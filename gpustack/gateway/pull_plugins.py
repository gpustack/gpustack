# This script is modified from https://raw.githubusercontent.com/higress-group/plugin-server/db204afc52e59c32c064d2c75b39bbe0a8b5b39c/pull_plugins.py
import os
import subprocess
import json
import argparse
import tarfile
import shutil
import hashlib
from typing import Optional
from datetime import datetime
from gpustack.gateway.plugins import (
    supported_plugins,
    get_plugin_url_with_name_and_version,
    get_wasm_plugin_dir,
)


def calculate_md5(file_path, chunk_size=4096):
    """Calculate the MD5 value of a file"""
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def handle_tar_layer(tar_path, target_dir):
    """
    Handle tar.gzip layer
    Return whether a wasm file is found
    """
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            wasm_files = [f for f in tar.getmembers() if f.name.endswith('.wasm')]
            if wasm_files:
                wasm_file = wasm_files[0]
                tar.extract(wasm_file, path=target_dir)
                old_path = os.path.join(target_dir, wasm_file.name)
                new_path = os.path.join(target_dir, 'plugin.wasm')
                os.rename(old_path, new_path)
                print(f"Successfully extracted .wasm file: {new_path}")
                return True
            else:
                print("No .wasm file found")
                return False
    except Exception as e:
        print(f"Error extracting tar file: {e}")
        return False


def handle_wasm_layer(wasm_path, target_dir):
    """
    Handle .wasm layer
    Return whether the wasm file was successfully copied
    """
    try:
        new_path = os.path.join(target_dir, 'plugin.wasm')
        shutil.copy2(wasm_path, new_path)
        print(f"Successfully copied .wasm file: {new_path}")
        return True
    except Exception as e:
        print(f"Error copying .wasm file: {e}")
        return False


def generate_metadata(plugin_dir, plugin_name):
    """
    Generate metadata.txt for plugin.wasm
    """
    wasm_path = os.path.join(plugin_dir, 'plugin.wasm')
    try:
        stat_info = os.stat(wasm_path)
        size = stat_info.st_size
        mtime = datetime.fromtimestamp(stat_info.st_mtime).isoformat()
        ctime = datetime.fromtimestamp(stat_info.st_ctime).isoformat()
        md5_value = calculate_md5(wasm_path)
        metadata_path = os.path.join(plugin_dir, 'metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"Plugin Name: {plugin_name}\n")
            f.write(f"Size: {size} bytes\n")
            f.write(f"Last Modified: {mtime}\n")
            f.write(f"Created: {ctime}\n")
            f.write(f"MD5: {md5_value}\n")
        print(f"Successfully generated metadata.txt: {metadata_path}")
    except Exception as e:
        print(f"Failed to generate metadata: {e}")


def main():
    parser = argparse.ArgumentParser(description='Process plugin configuration file')
    parser.add_argument(
        '--base-path', type=str, default=None, help='Base path to store plugins'
    )
    args = parser.parse_args()

    base_path: Optional[str] = getattr(args, 'base_path', None)
    if base_path is None:
        base_path = get_wasm_plugin_dir(True)
    if base_path is None:
        print("Failed to determine base path for plugins.")
        return

    failed_plugins = []

    for plugin in supported_plugins:
        if plugin.digest is None:
            continue
        plugin_url = get_plugin_url_with_name_and_version(
            plugin.name, plugin.version
        ).removeprefix("oci://")

        print(f"\nProcessing plugin: {plugin.name} {plugin.version}({plugin.digest})")
        success = process_plugin(base_path, plugin.name, plugin_url, plugin.version)
        if not success:
            failed_plugins.append(f"{plugin.name}:{plugin.version}")

    if failed_plugins:
        print("\nThe following plugins were not processed successfully:")
        for plugin in failed_plugins:
            print(f"- {plugin}")
        exit(1)


def process_plugin(plugins_base_path, plugin_name, plugin_url, version):
    """
    Process downloading and information retrieval for a single plugin
    """
    os.makedirs(plugins_base_path, exist_ok=True)

    plugin_dir = os.path.join(plugins_base_path, plugin_name, version)
    os.makedirs(plugin_dir, exist_ok=True)

    temp_download_dir = os.path.join(plugins_base_path, f"{plugin_name}_{version}_temp")
    os.makedirs(temp_download_dir, exist_ok=True)

    wasm_found = False

    try:
        subprocess.run(
            ['oras', 'cp', plugin_url, '--to-oci-layout', temp_download_dir], check=True
        )

        with open(os.path.join(temp_download_dir, 'index.json'), 'r') as f:
            index = json.load(f)

        manifest_digest = index['manifests'][0]['digest']
        manifest_path = os.path.join(
            temp_download_dir, 'blobs', 'sha256', manifest_digest.split(':')[1]
        )

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        for layer in manifest.get('layers', []):
            media_type = layer.get('mediaType', '')
            digest = layer.get('digest', '').split(':')[1]

            if media_type in [
                'application/vnd.docker.image.rootfs.diff.tar.gzip',
                'application/vnd.oci.image.layer.v1.tar+gzip',
            ]:
                tar_path = os.path.join(temp_download_dir, 'blobs', 'sha256', digest)
                wasm_found = handle_tar_layer(tar_path, plugin_dir)

            elif media_type == 'application/vnd.module.wasm.content.layer.v1+wasm':
                wasm_path = os.path.join(temp_download_dir, 'blobs', 'sha256', digest)
                wasm_found = handle_wasm_layer(wasm_path, plugin_dir)

    except subprocess.CalledProcessError as e:
        print(f"{plugin_name} ({version}) command execution failed: {e}")
        return False
    except Exception as e:
        print(f"Error occurred while processing {plugin_name} ({version}): {e}")
        return False
    finally:
        shutil.rmtree(temp_download_dir, ignore_errors=True)

    if wasm_found:
        generate_metadata(plugin_dir, plugin_name)

    return wasm_found


if __name__ == '__main__':
    main()
