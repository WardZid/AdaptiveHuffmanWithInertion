import random
import string
import os


def save_output(file_name, content):
    dir_path = os.path.dirname(__file__)

    out_path = os.path.join(dir_path, "out")

    # create out if non existent
    os.makedirs(out_path, exist_ok=True)

    # increment naming
    base_name, ext = os.path.splitext(file_name)
    i = 1
    while True:
        file_path = os.path.join(out_path, f"{base_name} ({i}){ext}" if i > 1 else file_name)
        if not os.path.exists(file_path):
            break
        i += 1

    with open(file_path, 'w') as f:
        f.write(content)


def generate_block(block_size, Y):
    word = ''.join(random.choice(string.ascii_lowercase) for _ in range(Y))
    repetitions = block_size // Y
    return word * repetitions


def generate_text(total_symbols, XXX, Y):
    word_length = Y
    num_of_blocks = XXX

    block_size = total_symbols // num_of_blocks

    final_text = ""
    for i in range(0, num_of_blocks):
        block = generate_block(block_size, word_length)
        final_text += block

    return final_text[:total_symbols]


text = generate_text(10000000, 944, 4)
# print(text)
print(len(text))
save_output("initial-text.txt", text)