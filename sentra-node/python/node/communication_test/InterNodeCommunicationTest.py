'''
This is a little test programm to test the inter Sentra node communication in combination with the Rust-based proxy functionality.
'''

import argparse
import socket
import threading
from time import sleep

def start_server(port:int):
    # Define server host and port
    host = '127.0.0.1'  # Localhost

    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the host and port
    server_socket.bind((host, port))

    # Start listening for incoming connections
    server_socket.listen(5)  # Allow up to 5 connections in the queue
    print(f"Server is listening on {host}:{port}...")

    while True:
        # Accept a new connection
        client_socket, client_address = server_socket.accept()
        print(f"Connection established with {client_address}")

        # Receive data from the client
        message = client_socket.recv(1024).decode('utf-8')  # Buffer size is 1024 bytes
        print(f"Received message: {message}")

        # Optionally, send a response back to the client
        client_socket.send("Message received!".encode('utf-8'))

        # Close the client connection
        client_socket.close()

# Function to start the server in a background thread
def start_server_in_background(port:int):
    server_thread = threading.Thread(target=start_server, daemon=True,args={port})
    server_thread.start()
    print("Server is running in the background...")


def start_client(port:int):
    # Define server host and port
    host = '127.0.0.1'  # Server's IP address

    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the server
    client_socket.connect((host, port))
    print(f"Connected to server at {host}:{port}")

    # Send a message to the server
    message = "Hello, Server! This is a test message."
    client_socket.send(message.encode('utf-8'))
    print(f"Sent message: {message}")

    # Receive a response from the server
    response = client_socket.recv(1024).decode('utf-8')
    print(f"Response from server: {response}")

    # Close the connection
    client_socket.close()

if __name__ == "__main__":
    m_Parser=argparse.ArgumentParser("Simple inter Sentra Node communication test")
    m_Parser.add_argument("-p","--port",help="Port to listen on for requests from other Sentra Nodes",default=8888)
    m_Parser.add_argument("--base_port",help="Base Port for communication with other clients. The base port is incremented for communication with each new Sentra node",default=8888)
    m_Parser.add_argument("--nr_nodes",help="number of Sentra nodes to communcate with",default=1)
    m_Parser.add_argument("--own_id",help="number of own Sentra node ID",default=-1)
    m_Args=m_Parser.parse_args()
    start_server_in_background(m_Args.port)
    i:int=0
    max_clients:int=m_Args.nr_nodes
    base_port:int=m_Args.base_port
    own_id:int=m_Args.own_id
    while True:
        if(i!=own_id):
            start_client(base_port+i)
            sleep(1)
        i=i+1
        if (i>=max_clients):
            i=0
