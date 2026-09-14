use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc;
use std::thread;

pub struct TCPProxy {
    port: u16,
}

impl TCPProxy {
    // Constructor to create a new TcpServer with a specified port
    pub fn new(port: u16) -> Self {
        TCPProxy { port }
    }

    // Function to start the server in a separate thread
    pub fn start(self, sender: mpsc::Sender<Vec<u8>>) {
        let port = self.port;

        // Spawn a new thread for the server
        thread::spawn(move || {
            // Use Tokio runtime for async operations
            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime.block_on(async move {
                // Bind the TcpListener to the specified port
                let addr = format!("0.0.0.0:{}", port);
                let listener = TcpListener::bind(&addr).await.unwrap();
                println!("Server listening on {}", addr);

                loop {
                    // Accept a single incoming connection
                    match listener.accept().await {
                        Ok((socket, _)) => {
                            let sender = sender.clone();

                            // Handle the single connection
                            Self::handle_connection(socket, sender).await;
                        }
                        Err(e) => {
                            println!("Failed to accept connection: {}", e);
                        }
                    }
                }
            });
        });
    }

    // Function to handle the single client connection
    async fn handle_connection(mut socket: TcpStream, sender: mpsc::Sender<Vec<u8>>) {
        let mut buffer = vec![0; 1024];

        loop {
            // Read data from the socket
            match socket.read(&mut buffer).await {
                Ok(0) => {
                    // Connection closed
                    println!("Connection closed");
                    break;
                }
                Ok(n) => {
                    // Send the received data through the channel
                    if sender.send(buffer[..n].to_vec()).await.is_err() {
                        println!("Failed to send data through the channel");
                        break;
                    }
                }
                Err(e) => {
                    println!("Failed to read from socket: {}", e);
                    break;
                }
            }
        }
    }

    // Function to send data to the connected client
    pub async fn send_to_client(&self, mut socket: TcpStream, data: &[u8]) {
        if let Err(e) = socket.write_all(data).await {
            println!("Failed to send data to the client: {}", e);
        }
    }
}

/* 
[tokio::main]
async fn main() {
    // Create a channel for communication
    let (tx, mut rx) = mpsc::channel(100);

    // Create a new TcpServer instance
    let server = TcpServer::new(8080);

    // Start the server
    server.start(tx);

    // Example: Process data received from the channel
    while let Some(data) = rx.recv().await {
        println!("Received data: {:?}", data);
    }
}
*/
