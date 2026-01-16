use ort::session::Session;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ENCODER MODEL ===");
    let encoder = Session::builder()?.commit_from_file("canary-1b-v2/encoder-model.onnx")?;

    println!("\nInputs:");
    for input in encoder.inputs().iter() {
        println!("  Name: {} ({})", input.name(), input.dtype());
    }

    println!("\nOutputs:");
    for output in encoder.outputs().iter() {
        println!("  Name: {} ({})", output.name(), output.dtype());
    }

    println!("\n=== DECODER MODEL ===");
    let decoder = Session::builder()?.commit_from_file("canary-1b-v2/decoder-model.onnx")?;

    println!("\nInputs:");
    for input in decoder.inputs().iter() {
        println!("  Name: {} ({})", input.name(), input.dtype());
    }

    println!("\nOutputs:");
    for output in decoder.outputs().iter() {
        println!("  Name: {} ({})", output.name(), output.dtype());
    }

    Ok(())
}
