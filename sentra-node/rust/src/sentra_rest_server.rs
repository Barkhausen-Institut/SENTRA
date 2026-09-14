/* 
use crate::sentra_attester;

use axum::{
    routing::get,
    Router,
    http::StatusCode,
    response::IntoResponse,
};



// Handler that returns binary response
async fn attest_handler() -> impl IntoResponse {
    let attestation_report = sentra_attester::generate_attestation_report();
    
    (
        StatusCode::OK,
        [("Content-Type", "application/octet-stream")],
        attestation_report
    )
}

//#[tokio::main]
pub async fn start_rest_server() {
    let app = Router::new()
        .route("/attest", get(attest_handler));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .unwrap();
    
    println!("Server running on http://0.0.0.0:3000");
    axum::serve(listener, app).await.unwrap();
}
*/