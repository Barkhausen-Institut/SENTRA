use tonic::{transport::Server,Request,Response,Status};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::protos::sentra_inter_node_grpc_services::InterNodeMessage;
use crate::protos::sentra_inter_node_grpc_services::inter_node_message::MessageType;
use crate::protos::sentra_inter_node_grpc_services::inter_node_message_service_server;
use crate::SentraNode;

#[derive(Default)]
pub struct SentraInterNodeMessageService
{
   node_id:String
}

#[tonic::async_trait]
impl inter_node_message_service_server::InterNodeMessageService for SentraInterNodeMessageService
{
    type InterNodeStreamStream = ReceiverStream<Result<InterNodeMessage, Status>>;

    async fn inter_node_stream(&self,request: Request<tonic::Streaming<InterNodeMessage>>) -> Result<Response<Self::InterNodeStreamStream>, Status> 
    {
        println!("New bidirectional inter node meassage stream started! - We are the server");
        let mut incoming_stream = request.into_inner();

            // Create a channel to send messages back to the client
        let (tx, rx) = mpsc::channel(32);


        // Spawn a task to handle incoming messages
        let node_id:String=self.node_id.clone();
        tokio::spawn(async move {
            while let result = incoming_stream.message().await {
                match result {
                    Ok(Some(message)) => 
                    {
                         handle_node_message(node_id.clone(),message)
                    }
                     Ok(None) =>
                                {
                                    // Stream ended
                                    println!("Stream closed by node");
                                    break;
                                }
                    Err(e) => {
                        eprintln!("Error receiving message: {}", e);
                        break;
                    }
                }
            }
            println!("Client disconnected or stream ended");
        });

        // Return the receiver stream as the response
        Ok(Response::new(ReceiverStream::new(rx)))
	}
} 

pub fn handle_node_message(this_node:String,node_msg: InterNodeMessage)
    {
        match node_msg.message_type 
        {
            Some(MessageType::Hello(hello)) => 
            {
                println!("Received Hello-Message from {} to {}...",hello.node_id,this_node);
	    },
            Some(MessageType::Heartbeat(heartbeat)) => 
            {
                println!("Received Heartbeat Message...");
	    },
	    None => {
        	println!("Received empty inter node message");
    	    }
        }
    }

async fn run_server(node_id:String) -> Result<(), Box<dyn std::error::Error>> {
    let addr = "0.0.0.0:50051".parse().unwrap();
    let mut inter_node_grpc_service = SentraInterNodeMessageService::default();
    inter_node_grpc_service.node_id=node_id;

    println!("Sentra inter node grpc service listening on {}", addr);

    Server::builder()
        .add_service(inter_node_message_service_server::InterNodeMessageServiceServer::new(inter_node_grpc_service))
        .serve(addr)
        .await?;

    Ok(())
}

pub fn startGRPCServer(sentra_node:&SentraNode)
{
    let node_id=sentra_node.node_id.clone();
    let server_thread = std::thread::spawn(|| {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            if let Err(e) = run_server(node_id).await {
                eprintln!("Server error: {}", e);
            }
        });
    });
}