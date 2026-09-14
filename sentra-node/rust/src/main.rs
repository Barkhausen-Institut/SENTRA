mod sentra_attester;
mod acme;

use std::{error::Error, time::Duration,process::Command,net::ToSocketAddrs};

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::{Channel, Certificate};
use hostname;
use rustls::crypto::{aws_lc_rs, CryptoProvider};
use dns_lookup::lookup_addr;
use rustc_version_runtime;

mod protos;
use crate::protos::sentra_backend_grpc_services::ServerMessage;
use crate::protos::sentra_backend_grpc_services::RegisterRequest;
use crate::protos::sentra_backend_grpc_services::NodeMessage;
use crate::protos::sentra_backend_grpc_services::node_message_service_client::NodeMessageServiceClient;
use crate::protos::sentra_backend_grpc_services::TGrpcSentraNode;
use crate::protos::sentra_backend_grpc_services::server_message;
use crate::protos::sentra_backend_grpc_services::node_message;
use crate::protos::sentra_backend_grpc_services::AttestationResponse;

mod command_line_options;
mod committee;
mod sentra_node;
mod sentra_node_grpc;

const SENTRA_NODE_VERSION: &str =env!("CARGO_PKG_VERSION");

use sentra_node::SentraNode;

impl Default for SentraNode {
    fn default()->Self
      {
        SentraNode
        {
            node_id: String::new(),
            grpc_url: String::new(),
            args: argh::from_env(),
            committee: committee::Committee::default()
        }
      }
}

impl SentraNode
{
    fn startPythonCode(&self)
    {
        let child = Command::new("/bin/python3")
        .arg("/bin/sentra/InterNodeCommunicationTest.py") 
        .spawn() // Spawns the process without waiting for it to finish
        .expect("Failed to start Python script");
        println!("Python script is running in the background!");
    }

    fn handle_join_committee_message(&self,committee:&Vec<TGrpcSentraNode>)
    {
        for node in committee
        {
            self.committee.add(&node.node_id,&node.grpc_url);
        }
        println!("{}",self.committee);
        self.committee.establish_connections();
        //self.startPythonCode();
    }

    fn handle_server_message(&self,server_msg: ServerMessage,tx:mpsc::Sender<NodeMessage>,fake_attestation:bool) {
    match server_msg.message_type {
        Some(server_message::MessageType::Response(ack)) => {
            println!("✓ ACK: {} - {}", ack.success, ack.message);
        },
        Some(server_message::MessageType::Attestation(req)) => {
            println!("Attestation request with nonce: {}...",req.nonce);
            println!("Spwan send quote thread...");
            let tx_clone: mpsc::Sender<NodeMessage>=tx;
	        tokio::spawn(async move {
                // Send attestatin message
                let quote: Vec<u8>=sentra_attester::generate_attestation_report(fake_attestation);
                println!("Sending attestation...");
                let register_msg: NodeMessage = NodeMessage {
                    message_type: Some(node_message::MessageType::Quote(AttestationResponse {
                        report: quote
                    }))
                };

                if tx_clone.send(register_msg).await.is_err() {
                    eprintln!("Failed to send attestation");
                    return;
                }
            });
        },
        Some(server_message::MessageType::PythonMsg(_)) => {
            println!("Some(server_message::MessageType::PythonMsg(_)) - not implemented");
        }
        Some(server_message::MessageType::JoinCommitteeRequest(request)) => {
            println!("Received JoinCommittee request -- Committee: {:?}", request.committee);
            self.handle_join_committee_message(&request.committee);
        },

        None => {
            println!("Received empty message");
        }
    }
    }
}

fn get_fqdn(hostname: &str) -> Option<String> {
    // Append a dummy port to the hostname for DNS resolution
    let addr = (hostname, 0);
    
    // Resolve the hostname to socket addresses
    if let Ok(mut addrs) = addr.to_socket_addrs() {
        // If resolution succeeds, return the first resolved address as a string
        if let Some(resolved_addr) = addrs.next() {
            return lookup_addr(&resolved_addr.ip()).ok();
        }
    }
    
    // Return None if resolution fails
    None
}

fn main()
{
    println!("Starting Sentra Node version: {} [compiled using {:?}]",SENTRA_NODE_VERSION,rustc_version_runtime::version());
    let mut sentra_node:SentraNode=SentraNode::default();

    sentra_node.node_id=match hostname::get()
        {
            Ok(name) =>
                {
                    let hname: String=name.to_string_lossy().to_string();
                    println!("Hostname: {}", hname);
                    hname
                }
            Err(e) =>
                {
                    eprintln!("Failed to get hostname: {}", e);
                    String::from("Unknown")
                }
        };
    let fqdn:String=match get_fqdn(&sentra_node.node_id) {
        Some(fqdn) => fqdn,
        None => sentra_node.node_id.clone()
    };
    sentra_node.grpc_url="http://".to_owned()+&fqdn+":50051";
    sentra_node.committee.set_this_node_id(&sentra_node.node_id);
    CryptoProvider::install_default(aws_lc_rs::default_provider()).expect("Failed to install crypto provider");
    let mut grpc_cert:Option<String>=None;
    let mut _grpc_key:Option<String>=None;
    if sentra_node.args.use_acme
    {
        match acme::get_tls_certificate(&sentra_node.args.acme_url,&sentra_node.args.acme_cert,&sentra_node.node_id)
            {
                Ok((cert, key)) =>
                    {
                        println!("Got certificate!");
                        grpc_cert=Some(cert);
                        _grpc_key=Some(key);
                    }
                Err(e) =>
                    {
                        eprintln!("MAIN ERROR: {}", e);
                        eprintln!("Full error chain:");
                        let mut current: Option<&dyn Error> = e.source();
                        while let Some(cause) = current
                            {
                                eprintln!("Caused by: {}", cause);
                                current = cause.source();
                            }
                        return;
                    }
        };
    }

    sentra_node_grpc::startGRPCServer(&sentra_node);

    let grpc_server_url: String=sentra_node.args.grpc_url.clone();

    let rt: tokio::runtime::Runtime = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async
        {
            // Connect to the server
            println!("Try to connect to GRPC interface of Sentra backend: {}",grpc_server_url);
            let mut client: NodeMessageServiceClient<tonic::transport::Channel>;
            if sentra_node.args.use_acme
            {
                let ca_cert = Certificate::from_pem(grpc_cert.unwrap().as_bytes());
                println!("Loaded CA certificate");

                let endpoint = Channel::from_shared(grpc_server_url).expect("REASON-1")
                    .tls_config(
                                tonic::transport::ClientTlsConfig::new()
                                .ca_certificate(ca_cert)
                                .domain_name("sentra-backend")
                                ).expect("REASON-2");

                let channel: Channel = endpoint.connect().await.expect("REASON-3");
                client = NodeMessageServiceClient::new(channel);
            }
            else
            {
                println!("Doing a default connection...");
                client = NodeMessageServiceClient::connect(grpc_server_url).await.expect("REASON");
            }

            // Create a channel for sending messages to the server
            println!("Create channels...");
            let (tx, rx) = mpsc::channel(32);

            // Create the stream from the receiver
            println!("Create outbound stream...");
            let outbound: ReceiverStream<NodeMessage> = ReceiverStream::new(rx);

            let tx_register: mpsc::Sender<NodeMessage>=tx.clone();
            println!("Spawn registration thread...");
            let node_id:String=sentra_node.node_id.clone();
            let node_grpc_url:String=sentra_node.grpc_url.clone();
            tokio::spawn(async move
                {
                    // Send registration message
                    println!("Sending registration...");
                    let register_msg: NodeMessage = NodeMessage {
                        message_type: Some(node_message::MessageType::Register(RegisterRequest {
                            node_id: node_id,
                            node_grpc_url:node_grpc_url
                        }))
                    };

                    if tx_register.send(register_msg).await.is_err() {
                        eprintln!("Failed to send registration");
                        return;
                    }
                });

            tokio::time::sleep(Duration::from_millis(100)).await;

            // Start the bidirectional stream
            println!("Start bidirectional stream...");
            let response_stream: Result<tonic::Response<tonic::Streaming<ServerMessage>>, tonic::Status>
                 = client.node_stream(outbound).await;
            println!("Wait for inbound...");

            let mut inbound: tonic::Streaming<ServerMessage> = response_stream.expect("REASON").into_inner();

            // Receive messages from the server
            println!("Listening for server messages...");
            loop
                {
                    match inbound.message().await
                        {
                            Ok(Some(server_msg)) =>
                                {
                                    sentra_node.handle_server_message(server_msg,tx.clone(),sentra_node.args.fake_attestation);
                                }
                            Ok(None) =>
                                {
                                    // Stream ended
                                    println!("Stream closed by server");
                                    break;
                                }
                            Err(status) =>
                                {
                                    eprintln!("Error receiving GRPC message from backend: {}", status);
                                    break;
                                }
                        }
                }
            println!("Exiting the GRPC message receive from backend loop...");
        });
    }



