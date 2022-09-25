import os


project = "training2223"
user = os.getenv("USER")

file_name = "hello-world.txt"
greeting = "hello world"


def write_to_file(path: str, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)


write_to_file(file_name, greeting)
write_to_file(f"/p/{project}/{user}/{file_name}", greeting)
