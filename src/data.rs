use std::fs;
use std::io::Read;
use regex::Regex;

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
    println!("loading {} files", file_paths.len());

    for file_path in file_paths {
        let file = fs::File::open(file_path);
        
        match file {
            Ok(mut file) => {
                let mut content = String::new();
                match file.read_to_string(&mut content) {
                    Ok(_) => loaded_text.push_str(&format!("{}\n", content)),
                    Err(e) => eprintln!("Error reading file: {}", e),
                }
            }
            Err(e) => eprintln!("Error opening file: {}", e),
        }
    }

    println!("{} 'what' questions loaded", loaded_text.matches("what").count());
    println!("{} 'why' questions loaded", loaded_text.matches("why").count());
    println!("{} 'how' questions loaded\n", loaded_text.matches("how").count());

    loaded_text
}

pub fn reduce_spaces(text: &str) -> String {
    let re = Regex::new(r" {2,}").unwrap();
    re.replace_all(text, " ").to_string()
}
