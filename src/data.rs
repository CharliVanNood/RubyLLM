use std::fs;
use std::io::Read;

fn get_file_paths(folder: &str) -> Vec<String> {
    let mut paths = Vec::new();

    if let Ok(entries) = fs::read_dir(folder) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.is_file() {
                    if let Some(path_str) = path.to_str() {
                        paths.push(path_str.to_string());
                    }
                }
            }
        }
    } else {
        eprintln!("Error reading directory: {}", folder);
    }

    paths
}

pub fn load_data() -> String {
    let mut loaded_text = String::new();

    let file_paths = get_file_paths("data");

    for file_path in file_paths {
        let file = fs::File::open(file_path);
        
        match file {
            Ok(mut file) => {
                let mut content = String::new();
                match file.read_to_string(&mut content) {
                    Ok(_) => loaded_text.push_str(&format!("\n{}", content)),
                    Err(e) => eprintln!("Error reading file: {}", e),
                }
            }
            Err(e) => eprintln!("Error opening file: {}", e),
        }
    }
    
    loaded_text
}
