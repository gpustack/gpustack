import csv
import logging
import subprocess
import xml.etree.ElementTree as ET
from gpustack.detectors.base import GPUDetectExepction, GPUDetector
from gpustack.schemas.workers import (
    GPUCoreInfo,
    GPUDeviceInfo,
    GPUDevicesInfo,
    MemoryInfo,
    VendorEnum,
)
from gpustack.utils import platform
from gpustack.utils.command import is_command_available
from gpustack.utils.convert import safe_float, safe_int
from gpustack.utils.envs import is_docker_env

logger = logging.getLogger(__name__)


class NvidiaSMI(GPUDetector):
    def is_available(self) -> bool:
        return is_command_available("nvidia-smi")

    def gather_gpu_info(self) -> GPUDevicesInfo:
        # First try CSV format (normal flow)
        command = self._command_gather_gpu()
        results = self._run_command(command)
        if results is None:
            return []

        # Try to decode CSV results
        devices = self.decode_gpu_devices(results)

        # Check if we got insufficient permissions or N/A values (indicating potential MIG issues)
        if not self._should_try_xml_fallback(results, devices):
            return devices

        # Try XML format for MIG support
        xml_command = self._command_gather_gpu_xml()
        xml_results = self._run_command(xml_command)
        if xml_results is None:
            return []

        xml_devices = self.decode_gpu_devices_xml(xml_results)
        return xml_devices

    def _should_try_xml_fallback(self, csv_results, devices):
        """
        Check if we should try XML fallback based on CSV results.
        Returns True if CSV results contain insufficient permissions or N/A values.
        """
        if not csv_results:
            return True

        # Check for permission issues or N/A values in CSV output
        csv_lower = csv_results.lower()
        if (
            'insufficient permissions' in csv_lower or 'n/a' in csv_lower or not devices
        ):  # No devices parsed from CSV
            return True

        # Check if any device has invalid memory values (0 or very small)
        for device in devices:
            if device.memory.total <= 0:
                return True

        return False

    def decode_gpu_devices_xml(self, xml_result) -> GPUDevicesInfo:  # noqa: C901
        """
        Parse XML output from nvidia-smi -q -x to extract GPU and MIG device information.
        This method handles both regular GPUs and MIG instances in containers.

        Output XML example (Non-critical components have been hidden):
        <?xml version="1.0" ?>
        <!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v12.dtd">
        <nvidia_smi_log>
            <driver_version>575.57.08</driver_version>
            <cuda_version>12.9</cuda_version>
            <attached_gpus>1</attached_gpus>
            <gpu id="00000000:00:09.0">
                <mig_mode>
                    <current_mig>Enabled</current_mig>
                    <pending_mig>Enabled</pending_mig>
                </mig_mode>
                <mig_devices>
                <mig_device>
                    <index>0</index>
                    <gpu_instance_id>1</gpu_instance_id>
                    <compute_instance_id>0</compute_instance_id>
                    <device_attributes>
                        <shared>
                            <multiprocessor_count>60</multiprocessor_count>
                            <copy_engine_count>3</copy_engine_count>
                            <encoder_count>0</encoder_count>
                            <decoder_count>3</decoder_count>
                            <ofa_count>0</ofa_count>
                            <jpg_count>3</jpg_count>
                        </shared>
                    </device_attributes>
                    <fb_memory_usage>
                        <total>40448 MiB</total>
                        <reserved>0 MiB</reserved>
                        <used>298 MiB</used>
                        <free>40151 MiB</free>
                    </fb_memory_usage>
                </mig_device>
                <temperature>
                    <gpu_temp>38 C</gpu_temp>
                    <gpu_temp_tlimit>45 C</gpu_temp_tlimit>
                </temperature>
                <fb_memory_usage>
                    <total>81920 MiB</total>
                    <reserved>446 MiB</reserved>
                    <used>3703 MiB</used>
                    <free>77771 MiB</free>
                </fb_memory_usage>
            </gpu>
        </nvidia_smi_log>
        """
        devices = []

        try:
            root = ET.fromstring(xml_result)

            # Find all GPU elements
            for gpu_elem in root.findall('.//gpu'):
                # Get basic GPU information
                gpu_index = safe_int(gpu_elem.get('id', '0'))
                product_name = self._get_xml_text(
                    gpu_elem, 'product_name', 'Unknown GPU'
                )
                uuid = self._get_xml_text(gpu_elem, 'uuid', '')

                # Get temperature
                temperature = safe_float(
                    self._get_xml_text(gpu_elem, 'temperature/gpu_temp', '0').replace(
                        ' C', ''
                    )
                )

                # Check if MIG is enabled
                mig_mode = self._get_xml_text(
                    gpu_elem, 'mig_mode/current_mig', 'Disabled'
                )

                if mig_mode.lower() == 'enabled':
                    # Handle MIG instances
                    mig_devices = self._parse_mig_instances(
                        gpu_elem, gpu_index, temperature
                    )
                    devices.extend(mig_devices)
                else:
                    # Handle regular GPU
                    device = self._parse_regular_gpu_xml(
                        gpu_elem, gpu_index, product_name, uuid, temperature
                    )
                    if device:
                        devices.append(device)

        except ET.ParseError as e:
            # If XML parsing fails, return empty list to fallback to CSV
            logger.error(f"Failed to parse GPU devices XML: {e}")
            return []
        except Exception as e:
            # Log error but don't crash, fallback to CSV
            logger.error(f"Error in decode_gpu_devices_xml: {e}")
            return []

        return devices

    def _get_xml_text(self, element, path, default=''):
        """Helper method to safely get text from XML element."""
        try:
            elem = element.find(path)
            return elem.text if elem is not None and elem.text is not None else default
        except Exception:
            return default

    def _parse_mig_instances(self, gpu_elem, gpu_index, gpu_temperature):
        """Parse MIG instances from GPU XML element."""
        devices = []

        # Get MIG device names from nvidia-smi commands
        mig_device_names, is_privileges = self._get_mig_device_names()

        # Look for MIG devices in the XML structure
        mig_devices_elem = gpu_elem.find('mig_devices')
        if mig_devices_elem is not None:
            device_index = 0
            for mig_device in mig_devices_elem.findall('mig_device'):
                # Get MIG instance information
                gi_id = safe_int(self._get_xml_text(mig_device, 'gpu_instance_id', '0'))
                ci_id = safe_int(
                    self._get_xml_text(mig_device, 'compute_instance_id', '0')
                )
                mig_index = safe_int(self._get_xml_text(mig_device, 'index', '0'))

                # Get multiprocessor count
                mp_count = safe_int(
                    self._get_xml_text(
                        mig_device, 'device_attributes/shared/multiprocessor_count', '0'
                    )
                )

                # Try to get profile name from nvidia-smi commands
                target_pair = (gi_id, ci_id) if is_privileges else (mig_index, 0)
                profile_name = mig_device_names.get(target_pair) or "MIG Device"
                # Build device name using loose structure - only include available fields
                device_name_parts = []
                # Add GPU info if available
                if gpu_index is not None:
                    device_name_parts.append(f"GPU {gpu_index}")

                # Add GI info if available and valid
                if gi_id is not None:
                    device_name_parts.append(f"GI {gi_id}")

                # Add CI info if available and valid
                if ci_id is not None:
                    device_name_parts.append(f"CI {ci_id}")

                # Join parts with appropriate separators
                device_name = profile_name + "(" + "/".join(device_name_parts) + ")"

                # Get memory information from MIG device
                memory_total = (
                    safe_int(
                        self._get_xml_text(
                            mig_device, 'fb_memory_usage/total', '0'
                        ).replace(' MiB', '')
                    )
                    * 1024
                    * 1024
                )
                memory_used = (
                    safe_int(
                        self._get_xml_text(
                            mig_device, 'fb_memory_usage/used', '0'
                        ).replace(' MiB', '')
                    )
                    * 1024
                    * 1024
                )

                # MIG devices typically don't report GPU utilization in XML
                utilization_gpu = safe_float(
                    self._get_xml_text(
                        mig_device, 'utilization/gpu_util', '0.00'
                    ).replace(' %', '')
                )

                device = GPUDeviceInfo(
                    index=device_index,
                    device_index=device_index,
                    device_chip_index=0,
                    name=device_name,
                    vendor=VendorEnum.NVIDIA.value,
                    memory=MemoryInfo(
                        is_unified_memory=False,
                        used=memory_used,
                        total=memory_total,
                        utilization_rate=(
                            (memory_used / memory_total) * 100
                            if memory_total > 0
                            else 0
                        ),
                    ),
                    core=GPUCoreInfo(
                        utilization_rate=utilization_gpu,
                        total=mp_count,  # Use multiprocessor count as core total
                    ),
                    temperature=gpu_temperature,  # Use parent GPU temperature
                    type=platform.DeviceTypeEnum.CUDA.value,
                )
                devices.append(device)
                device_index += 1

        return devices

    def _get_mig_device_names(self):
        """
        Get MIG device names from nvidia-smi commands.
        First try nvidia-smi mig -lci (privileged), then fallback to nvidia-smi -L (unprivileged).
        Returns a dict mapping (gi_id, ci_id) to device name.
        """
        device_names = {}
        is_privileges = True
        # First try nvidia-smi mig -lci (requires privileges)
        try:
            lci_command = ["nvidia-smi", "mig", "-lci"]
            lci_result = self._run_command(lci_command)
            if lci_result and "insufficient permissions" not in lci_result.lower():
                device_names = self._parse_mig_lci_output(lci_result)
                if device_names:
                    return device_names, is_privileges
        except Exception as e:
            logger.trace(f"Failed to run nvidia-smi mig -lci: {e}")

        is_privileges = False
        # Fallback to nvidia-smi -L (unprivileged)
        try:
            list_command = ["nvidia-smi", "-L"]
            list_result = self._run_command(list_command)
            if list_result:
                device_names = self._parse_nvidia_smi_list_output(list_result)
        except Exception as e:
            logger.trace(f"Failed to run nvidia-smi -L: {e}")

        return device_names, is_privileges

    def _parse_mig_lci_output(self, output):
        """
        Parse nvidia-smi mig -lci table output to extract device names.
        Example format:
        +--------------------------------------------------------------------+
        | Compute instances:                                                 |
        | GPU     GPU       Name             Profile   Instance   Placement  |
        |       Instance                       ID        ID       Start:Size |
        |         ID                                                         |
        |====================================================================|
        |   0      1       MIG 3g.40gb          2         0          0:4     |
        +--------------------------------------------------------------------+
        |   0      2       MIG 3g.40gb          2         0          0:4     |
        +--------------------------------------------------------------------+
        """
        device_names = {}
        lines = output.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('+') or line.startswith('|='):
                continue
            if 'GPU' in line and 'Instance' in line and 'Name' in line:
                continue

            if line.startswith('|') and line.endswith('|'):
                # Remove the leading and trailing '|'
                content = line[1:-1].strip()
                # Split by whitespace and clean up
                parts = content.split()
                if len(parts) >= 7:  # Expecting at least 7 fields based on the format
                    try:
                        gi_id = safe_int(parts[1])
                        name = parts[2] + " " + parts[3]  # "MIG 3g.40gb"
                        ci_id = safe_int(parts[5])
                        device_names[(gi_id, ci_id)] = name
                    except Exception:
                        continue

        return device_names

    def _parse_nvidia_smi_list_output(self, output):
        """
        Parse nvidia-smi -L output to extract MIG device names.
        Example format:
        GPU 0: NVIDIA A100-PCIE-40GB (UUID: GPU-xxx)
          MIG 1g.5gb      Device  0: (UUID: MIG-xxx)
          MIG 1g.5gb      Device  1: (UUID: MIG-xxx)
        """
        device_names = {}
        lines = output.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('MIG ') and 'Device' in line:
                # Extract MIG profile name and device index
                # Format: "MIG 1g.5gb      Device  0: (UUID: MIG-xxx)"
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        profile_name = f"{parts[0]} {parts[1]}"  # "MIG 1g.5gb"
                        device_idx = safe_int(parts[3].rstrip(':'))
                        # For nvidia-smi -L, we don't have GI/CI mapping, so use device index
                        device_names[(device_idx, 0)] = profile_name
                    except Exception:
                        continue

        return device_names

    def _parse_regular_gpu_xml(
        self, gpu_elem, gpu_index, product_name, uuid, temperature
    ):
        """Parse regular GPU from XML element."""
        # Get memory information
        memory_total = (
            safe_int(
                self._get_xml_text(gpu_elem, 'fb_memory_usage/total', '0').replace(
                    ' MiB', ''
                )
            )
            * 1024
            * 1024
        )
        memory_used = (
            safe_int(
                self._get_xml_text(gpu_elem, 'fb_memory_usage/used', '0').replace(
                    ' MiB', ''
                )
            )
            * 1024
            * 1024
        )

        # Get utilization
        utilization_gpu = safe_float(
            self._get_xml_text(gpu_elem, 'utilization/gpu_util', '0').replace(' %', '')
        )

        device = GPUDeviceInfo(
            index=gpu_index,
            device_index=gpu_index,
            device_chip_index=0,
            name=product_name,
            vendor=VendorEnum.NVIDIA.value,
            memory=MemoryInfo(
                is_unified_memory=False,
                used=memory_used,
                total=memory_total,
                utilization_rate=(
                    (memory_used / memory_total) * 100 if memory_total > 0 else 0
                ),
            ),
            core=GPUCoreInfo(
                utilization_rate=utilization_gpu,
                total=0,  # Total cores information is not provided by nvidia-smi
            ),
            temperature=temperature,
            type=platform.DeviceTypeEnum.CUDA.value,
        )

        return device

    def decode_gpu_devices(self, result) -> GPUDevicesInfo:  # noqa: C901
        """
        results example:
        $nvidia-smi --format=csv,noheader --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu
        0, NVIDIA GeForce RTX 4080 SUPER, 16376 MiB, 1309 MiB, 0 %, 41
        1, NVIDIA GeForce RTX 4080 SUPER, 16376 MiB, 13625 MiB, 0 %, 39
        """

        devices = []
        reader = csv.reader(result.splitlines())
        for row in reader:
            if len(row) < 6:
                continue
            index, name, memory_total, memory_used, utilization_gpu, temperature_gpu = (
                row
            )

            index = safe_int(index)
            name = name.strip()
            # Convert MiB to bytes
            memory_total = safe_int(memory_total.split()[0]) * 1024 * 1024
            # Convert MiB to bytes
            memory_used = safe_int(memory_used.split()[0]) * 1024 * 1024
            utilization_gpu = safe_float(
                utilization_gpu.split()[0]
            )  # Remove the '%' sign
            temperature_gpu = safe_float(temperature_gpu)

            device = GPUDeviceInfo(
                index=index,
                device_index=index,
                device_chip_index=0,
                name=name,
                vendor=VendorEnum.NVIDIA.value,
                memory=MemoryInfo(
                    is_unified_memory=False,
                    used=memory_used,
                    total=memory_total,
                    utilization_rate=(
                        (memory_used / memory_total) * 100 if memory_total > 0 else 0
                    ),
                ),
                core=GPUCoreInfo(
                    utilization_rate=utilization_gpu,
                    total=0,  # Total cores information is not provided by nvidia-smi
                ),
                temperature=temperature_gpu,
                type=platform.DeviceTypeEnum.CUDA.value,
                runtime_framework="cuda",
            )
            devices.append(device)
        return devices

    def _run_command(self, command):
        result = None
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8"
            )

            if result is None or result.stdout is None:
                return None

            output = result.stdout
            if "no devices" in output.lower():
                return None

            if "Failed to initialize NVML: Unknown Error" in output and is_docker_env():
                raise GPUDetectExepction(
                    f"Error: {output}"
                    "Please ensure nvidia-smi is working properly. It may be caused by a known issue with the NVIDIA Container Toolkit, which can be mitigated by disabling systemd cgroup management in Docker. More info: <a href=\"https://docs.gpustack.ai/0.6/installation/nvidia-cuda/online-installation/?h=native.cgroupdriver=cgroupfs#prerequisites_1\">Disabling Systemd Cgroup Management in Docker</a>"
                )

            if result.returncode != 0:
                raise Exception(f"Unexpected return code: {result.returncode}")

            if output == "" or output is None:
                raise Exception(f"Output is empty, return code: {result.returncode}")

            return output
        except GPUDetectExepction as e:
            raise e
        except Exception as e:
            error_message = f"Failed to execute {command}: {e}"
            if result:
                error_message += f", stdout: {result.stdout}, stderr: {result.stderr}"
            raise Exception(error_message)

    def _command_gather_gpu(self):
        executable_command = [
            "nvidia-smi",
            "--format=csv,noheader",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
        ]
        return executable_command

    def _command_gather_gpu_xml(self):
        executable_command = [
            "nvidia-smi",
            "-q",
            "-x",  # XML format
        ]
        return executable_command
