use safetensors::SafeTensors;
use std::error::Error;
use std::fs;

fn main() -> Result<(), Box<dyn Error>> {
    let data = fs::read("./models/GPT2/model.safetensors")?;

    let st = SafeTensors::deserialize(&data)?;

    println!("Found {} tensors", st.len());

    for (name, view) in st.tensors() {
        println!(
            "{} -> shape: {:?}, dtype: {:?}",
            name,
            view.shape(),
            view.dtype()
        );
    }

    Ok(())
}
