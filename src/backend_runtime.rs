use crate::AppError;
use crate::AppResult;
use std::collections::HashMap;
use std::process::Command;

/// Runtime execution helper for command-style Candle/Burn adapters.
#[derive(Debug, Clone)]
pub struct RuntimeCommand {
    /// Executable path or program name.
    pub binary: String,
    /// Template args to pass on every run.
    pub args: Vec<String>,
}

impl RuntimeCommand {
    /// Executes the command after replacing placeholders with provided variables.
    pub fn run_with_vars(&self, vars: &HashMap<&str, &str>) -> AppResult<String> {
        let mut command = Command::new(&self.binary);

        for raw_arg in &self.args {
            command.arg(render_arg(raw_arg, vars));
        }

        let output = command.output().map_err(AppError::Io)?;

        if !output.status.success() {
            return Err(AppError::CommandFailure {
                command: self.binary.clone(),
                status: output.status.code().unwrap_or(-1),
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            });
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}

fn render_arg(arg: &str, vars: &HashMap<&str, &str>) -> String {
    let mut value = arg.to_string();

    for (key, value_ref) in vars {
        let placeholder = format!("{{{key}}}");
        value = value.replace(&placeholder, value_ref);
    }

    value
}
