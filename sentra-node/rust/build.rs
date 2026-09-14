fn main() -> Result<(), Box<dyn std::error::Error>> 
    {
        tonic_build::configure()
            .build_server(false)  // We only need the client    
	        .compile_protos(&["./SentraBackend-GRPC-Services.proto"],&["./"])?;
        tonic_build::configure()
            .build_server(true)     
	        .compile_protos(&["./SentraInterNode-GRPC-Services.proto"],&["./"])?;
        Ok(())
    }
