use crate::command_line_options;
use crate::committee;

pub struct SentraNode
{
    pub node_id: String,
    pub grpc_url: String,
    pub args: command_line_options::CommandLineOptions,
    pub committee: committee::Committee
}