from ...utils import run_periodically_async


class Agent:
    def __init__(self):
        self.registration_completed = False

    def start(self):
        """
        Start the agent.
        """

        # Report the node status to the server periodically.
        run_periodically_async(self.sync_node_status, self.interval)

        self.sync_loop()

    def sync_loop(self):
        """
        Main loop for processing changes. It watches task changes from server and processes them.
        """

        while True:
            pass

    def sync_node_status(self):
        """
        Should be called periodically to sync the node status with the server.
        It registers the node with the server if necessary.
        """

        self.register_with_server()
        self.update_node_status()

    def update_node_status(self):
        # 1. get node from server
        # 2. update node status if there is any change or enough time passed since last update

        pass

    def register_with_server(self):
        if self.registration_completed:
            return

        node = self.initial_node()
        self.register_node(node)
        self.registration_completed = True

    def register_node(self, node):
        # 1. create a node using the client
        # 2. if the node is already registered, update the node
        pass

    def initial_node(self):
        # initialize a node with the current system information
        pass
