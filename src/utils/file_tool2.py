# import 

# read file
def read_file_gen(file_path, to_be_split=" "):
    with open(file_path) as f:
        all_lines = f.readlines()
        for index, single_line in enumerate(all_lines):
            single_items = single_line.replace("\n", "").split(to_be_split)
            yield index, single_items
