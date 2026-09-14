use argh::FromArgs;

#[derive(FromArgs)]
/// Reach new heights.
pub struct CommandLineOptions {
    /// the Sentra Backend GRPC URL
    #[argh(option,default = "String::from(\"http://127.0.0.1/8000\")")]
    pub grpc_url: String,
    
    /// whether or not to use ACME for get node certificate
    #[argh(switch)]
    pub use_acme: bool,

    /// the ACME Server URL
    #[argh(option,default = "String::from(\"https://127.0.0.1/14000\")")]
    pub acme_url: String,

    /// the certificate file to be sued to establish a secure connetion to the ACME Server
    #[argh(option,default = "String::from(\"pebble.cer\")")]
    pub acme_cert: String,

    /// whether or not to fake the attestion i.e. sending a random attestation report (for easy testing outside an enclave)
    #[argh(switch)]
    pub fake_attestation: bool


}
