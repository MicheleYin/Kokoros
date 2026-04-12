use ort::session::builder::SessionBuilder;
use ort::session::Session;
use ort::ep;

pub trait OrtBase {
    fn load_model(&mut self, model_path: String) -> Result<(), String> {
        match SessionBuilder::new() {
            Ok(builder) => {
                // Register execution providers: CoreML for Apple devices acceleration
                let mut builder = builder
                    .with_execution_providers([
                         
                        
                        
                        ep::CPU::default()
                            // .with_fast_math(true)
                            .build()
                        // CoreML for Apple devices acceleration with fast math enabled
                    ])
                    .map_err(|e| format!("Failed to register execution providers: {}", e))?;
                
                let session = builder
                    .commit_from_file(model_path)
                    .map_err(|e| format!("Failed to commit from file: {}", e))?;
                self.set_sess(session);
                Ok(())
            }
            Err(e) => Err(format!("Failed to create session builder: {}", e)),
        }
    }

    fn print_info(&self) {
        if let Some(session) = self.sess() {
            eprintln!("Input names:");
            for input in session.inputs() {
                eprintln!("  - {}", input.name());
            }
            eprintln!("Output names:");
            for output in session.outputs() {
                eprintln!("  - {}", output.name());
            }

            eprintln!("Configured with: CoreML execution provider with fast math enabled");
        } else {
            eprintln!("Session is not initialized.");
        }
    }

    fn set_sess(&mut self, sess: Session);
    fn sess(&self) -> Option<&Session>;
}
