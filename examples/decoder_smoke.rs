use ort::session::Session;
use ort::tensor::{Shape, TensorElementType};
use ort::value::{DynTensor, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("CANARY_LOAD_ENCODER")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    {
        let _encoder = Session::builder()?.commit_from_file("canary-1b-v2/encoder-model.onnx")?;
    }

    let mut decoder = Session::builder()?.commit_from_file("canary-1b-v2/decoder-model.onnx")?;

    let prompt_len = std::env::var("CANARY_PROMPT_LEN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4);
    let mems_len = std::env::var("CANARY_MEMS_LEN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let encoded_len = std::env::var("CANARY_ENCODED_LEN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(10);

    let input_ids = vec![4_i64; prompt_len];
    let input_ids_tensor = Tensor::<i64>::from_array(([1, prompt_len], input_ids))?;

    let encoder_embeddings = vec![0.0f32; 1 * encoded_len * 1024];
    let encoder_embeddings_tensor =
        Tensor::<f32>::from_array(([1, encoded_len, 1024], encoder_embeddings))?;

    let encoder_mask = vec![1_i64; 1 * encoded_len];
    let encoder_mask_tensor = Tensor::<i64>::from_array(([1, encoded_len], encoder_mask))?;

    let mems_shape = Shape::new([10, 1, mems_len as i64, 1024]);
    let decoder_mems = DynTensor::new(decoder.allocator(), TensorElementType::Float32, mems_shape)?;

    eprintln!(
        "Running decoder smoke test (prompt_len={}, mems_len={}, encoded_len={})",
        prompt_len, mems_len, encoded_len
    );

    let outputs = decoder.run(ort::inputs![
        "input_ids" => input_ids_tensor,
        "encoder_embeddings" => encoder_embeddings_tensor,
        "encoder_mask" => encoder_mask_tensor,
        "decoder_mems" => decoder_mems,
    ])?;

    let logits = outputs["logits"].try_extract_tensor::<f32>()?;
    let hidden_states = outputs["decoder_hidden_states"].try_extract_tensor::<f32>()?;

    println!("logits shape: {:?}", logits.0.as_ref());
    println!(
        "decoder_hidden_states shape: {:?}",
        hidden_states.0.as_ref()
    );

    Ok(())
}
