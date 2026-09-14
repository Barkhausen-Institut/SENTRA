use std::time::Duration;
use std::fs;
use std::error::Error;

use instant_acme::{Account, ChallengeType, Identifier, NewAccount, NewOrder, OrderStatus};
use hyper_rustls::HttpsConnectorBuilder;
use hyper_util::client::legacy::Client;
use rustls::{ClientConfig, RootCertStore};
 
pub fn get_tls_certificate(acme_url:&String,acme_cert:&String,node_id:&String) -> Result<(String, String), Box<dyn std::error::Error>> 
    {
        let result: Result<(String, String), Box<dyn Error>>= tokio::runtime::Runtime::new()?.block_on(
            async 
                {
                    get_tlscertificate_async(acme_url,acme_cert,node_id).await
                });
        result
    }
    
pub async fn get_tlscertificate_async(acme_url:&String,acme_cert:&String,node_id:&String)-> Result<(String, String), Box<dyn std::error::Error>>
{
// Load root certificate
    let pem:Vec<u8>= fs::read(acme_cert)?;
    let certs: Vec<rustls::pki_types::CertificateDer<'_>> = rustls_pemfile::certs(&mut &pem[..])
        .collect::<Result<Vec<_>, _>>()?;
    
    let mut root_store: RootCertStore = RootCertStore::empty();
    for cert in certs 
        {
            root_store.add(cert)?;
        }
    
    // Build rustls config
    let tls_config: ClientConfig = ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    
    // Create HTTPS connector
    let https: hyper_rustls::HttpsConnector<hyper_util::client::legacy::connect::HttpConnector> = HttpsConnectorBuilder::new()
        .with_tls_config(tls_config)
        .https_only()
        .enable_http1()
        .build();
    
    // Create hyper client
    let http_client = Client::builder(hyper_util::rt::TokioExecutor::new())
        .build(https);

    // Create account
    let (account, _) = Account::create_with_http(
        &NewAccount {
            contact: &[],
            terms_of_service_agreed: true,
            only_return_existing: false,
        },
        &(acme_url.to_owned()+"/dir"),
        None,
        Box::new(http_client)
    )
    .await?;

    let params: rcgen::CertificateParams = rcgen::CertificateParams::new(vec![node_id.to_string()])?;
    let key_pair: rcgen::KeyPair = rcgen::KeyPair::generate()?;
    
    let private_key: String = key_pair.serialize_pem();
    let csr: rcgen::CertificateSigningRequest = params.serialize_request(&key_pair)?;
    let csr_der: Vec<u8> = csr.der().to_vec();
    // Create order
    let mut order: instant_acme::Order = account
        .new_order(&NewOrder {
            identifiers: &[Identifier::Dns(node_id.to_string())],
        })
        .await?;

    println!("Certificate Order placed...");
    println!("Fetching authorizations...");
    let authorizations = order.authorizations().await?;
    for authz in &authorizations 
        {
            let challenge = authz
                .challenges
                .iter()
                .find(|c| matches!(c.r#type, ChallengeType::Http01 ))
                .ok_or("No supported challenge type found")?;
    
            if challenge.r#type == ChallengeType::Http01 
                {
                    order.set_challenge_ready(&challenge.url).await?;
                }
        }

    tokio::time::sleep(Duration::from_millis(1000)).await;
    while order.state().status == OrderStatus::Pending 
        {
            tokio::time::sleep(Duration::from_secs(1)).await;
            order.refresh().await?;
        }
    
    // Check if ready
    if order.state().status != OrderStatus::Ready 
        {
            return Err(format!("Order not ready: {:?}", order.state()).into());
        }
    // Finalize with CSR (this waits for validation internally)
    println!("Try to finalise order...");
    order.finalize(&csr_der).await?;
    println!("Order finalised...");

    // Wait for certificate to be ready
    while order.state().status == OrderStatus::Processing 
        {
            tokio::time::sleep(Duration::from_secs(1)).await;
            order.refresh().await?;
        }

    // Get certificate
    match order.certificate().await? 
        {
            Some(cert_chain) => Ok((cert_chain, private_key)),
            None => Err("Certificate not ready".into())
        }
}
