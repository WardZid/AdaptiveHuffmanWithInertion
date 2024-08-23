import datetime
import operator
import os
import collections
import math
import csv

from bitarray import bitarray, bits2bytes


### UTIL METHODS
def bin_str2bool_list(binary_string):
    return [c == '1' for c in binary_string]


def bool_list2bin_str(boolean_list):
    return ''.join('1' if i else '0' for i in boolean_list)


def bool_list2int(boolean_list):
    return sum(v << i for i, v in enumerate(reversed(boolean_list)))


def entropy(byte_seq):
    counter = collections.Counter(byte_seq)
    ret = 0
    for count in counter.values():
        prob = count / sum(counter.values())
        ret += prob * math.log2(prob)
    return -ret


NYT = 'NYT'

# pylint: disable=too-many-instance-attributes
class Tree:
    def __init__(self, weight, num, data=None):
        """Use a set (`nodes`) to store all nodes in order to search the same
        weight nodes (block) iteratively which would be faster than recursive
        traversal of a tree.
        """
        self.weight = weight
        self.num = num
        self._left = None
        self._right = None
        self.parent = None
        self.data = data
        # code will not be always updated
        self.code = []

    def __repr__(self):
        return "#%d(%d)%s '%s'" % (self.num, self.weight, self.data, self.code)

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, left):
        self._left = left
        if self._left:
            self._left.parent = self

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, right):
        self._right = right
        if self._right:
            self._right.parent = self

    def pretty(self, indent_str='  '):
        return ''.join(self.pretty_impl(0, indent_str))

    def pretty_impl(self, level, indent_str):
        if not self._left and not self._right:
            return [indent_str * level, '%s' % self, '\n']
        line = [indent_str * level, '%s' % self, '\n']
        for subtree in (self._left, self._right):
            if isinstance(subtree, Tree):
                line += subtree.pretty_impl(level + 1, indent_str)
        return line

    def search(self, target):
        """Search a specific data according within the tree. Return the code of
        corresponding node if found. The code is the path from the root to the
        target node. If not found in the tree, return the code of NYT node.

        Args:
            target (any): The target data which needs to be found.

        Returns:
            {'first_appearance': bool, 'code': str}: An dictionary which
                contain the information of searching result.
        """

        stack = collections.deque([self])
        nytcode = stack[-1].code
        while stack:
            current = stack.pop()
            if current.data == target:
                return {'first_appearance': False, 'code': current.code}
            if current.data == NYT:
                nytcode = current.code
            if current.right:
                current.right.code = current.code + [1]
                stack.append(current.right)
            if current.left:
                current.left.code = current.code + [0]
                stack.append(current.left)
        return {'first_appearance': True, 'code': nytcode}


def exchange(node1, node2):
    """Exchange the children, data of two nodes but keep the number, parent and
    weight the same. Note that this function will not change the reference of
    `node1` and `node2`.
    """

    node1.left, node2.left = node2.left, node1.left
    node1.right, node2.right = node2.right, node1.right
    node1.data, node2.data = node2.data, node1.data


# pylint: disable=too-many-instance-attributes
class AdaptiveHuffman:
    def __init__(self, byte_seq, alphabet_range=(0, 255), INERTION=999999999):
        """Create an adaptive huffman encoder and decoder.

        Args:
            byte_seq (bytes): The bytes sequence to encode or decode.
            alphabet_range (tuple or integer): The range of alphabet
                inclusively.
            INERTION (int): The maximal number of sent letters before reducing weights.
        """

        self.byte_seq = byte_seq

        self._bits = None  # Only used in decode().
        self._bits_idx = 0  # Only used in decode().

        self.INERTION = INERTION
        self.symbols_count = 0  # track number of symbols processed

        # Get the first decimal number of all alphabets
        self._alphabet_first_num = min(alphabet_range)
        alphabet_size = abs(alphabet_range[0] - alphabet_range[1]) + 1
        # Select an `exp` and `rem` which meet `alphabet_size = 2**exp + rem`.
        # Get the largest `exp` smaller than `alphabet_size`.
        self.exp = alphabet_size.bit_length() - 1
        self.rem = alphabet_size - 2 ** self.exp

        # Initialize the current node # as the maximum number of nodes with
        # `alphabet_size` leaves in a complete binary tree.
        self.current_node_num = alphabet_size * 2 - 1

        self.tree = Tree(0, self.current_node_num, data=NYT)
        self.all_nodes = [self.tree]
        self.nyt = self.tree  # initialize the NYT reference

    def encode(self):
        """Encode the target byte sequence into compressed bit sequence by
        adaptive Huffman coding.

        Returns:
            bitarray: The compressed bitarray. Use `bitarray.tofile()` to save
                to file.
        """

        def encode_fixed_code(dec):
            """Convert a decimal number into specified fixed code.

            Arguments:
                dec {int} -- The alphabet need to be converted into fixed code.

            Returns:
                list of bool -- Fixed codes.
            """

            alphabet_idx = dec - (self._alphabet_first_num - 1)
            if alphabet_idx <= 2 * self.rem:
                fixed_str = '{:0{padding}b}'.format(
                    alphabet_idx - 1,
                    padding=self.exp + 1
                )
            else:
                fixed_str = '{:0{padding}b}'.format(
                    alphabet_idx - self.rem - 1,
                    padding=self.exp
                )
            return bin_str2bool_list(fixed_str)

        # print('entropy: %f', entropy(self.byte_seq))

        code = []
        for symbol in self.byte_seq:
            fixed_code = encode_fixed_code(symbol)
            result = self.tree.search(fixed_code)
            if result['first_appearance']:
                code.extend(result['code'])  # send code of NYT
                code.extend(fixed_code)  # send fixed code of symbol
            else:
                # send code which is path from root to the node of symbol
                code.extend(result['code'])
            self.update(fixed_code, result['first_appearance'])

        # Add remaining bits length info at the beginning of the code in order
        # to avoid the decoder regarding the remaining bits as actual data. The
        # remaining bits length info require 3 bits to store the length. Note
        # that the first 3 bits are stored as big endian binary string.
        remaining_bits_length = (
                bits2bytes(len(code) + 3) * 8 - (len(code) + 3)
        )
        code = (bin_str2bool_list('{:03b}'.format(remaining_bits_length))
                + code)

        return bitarray(code)

    def decode(self):
        """Decode the target byte sequence which is encoded by adaptive Huffman
        coding.

        Returns:
            list: A list of integer representing the number of decoded byte
                sequence.
        """

        def read_bits(bit_count):
            """Read n leftmost bits and move iterator n steps.

            Arguments:
                n {int} -- The # of bits is about to read.

            Returns:
                list -- The n bits has been read.
            """

            ret = self._bits[self._bits_idx:self._bits_idx + bit_count]
            self._bits_idx += bit_count
            return ret

        def decode_fixed_code():
            fixed_code = read_bits(self.exp)
            integer = bool_list2int(fixed_code)
            if integer < self.rem:
                fixed_code += read_bits(1)
                integer = bool_list2int(fixed_code)
            else:
                integer += self.rem
            return integer + 1 + (self._alphabet_first_num - 1)

        # Get boolean list ([True, False, ...]) from bytes.
        bits = bitarray()
        bits.frombytes(self.byte_seq)
        self._bits = bits.tolist()
        self._bits_idx = 0

        # Remove the remaining bits in the last of bit sequence generated by
        # bitarray.tofile() to fill up to complete byte size (8 bits). The
        # remaining bits length could be retrieved by reading the first 3 bits.
        # Note that the first 3 bits are stored as big endian binary string.
        remaining_bits_length = bool_list2int(read_bits(3))
        if remaining_bits_length:
            del self._bits[-remaining_bits_length:]
        self._bits = tuple(self._bits)

        code = []
        while self._bits_idx < len(self._bits):
            current_node = self.tree  # go to root
            while current_node.left or current_node.right:
                bit = read_bits(1)[0]
                current_node = current_node.right if bit else current_node.left
            if current_node.data == NYT:
                first_appearance = True
                dec = decode_fixed_code()
                code.append(dec)
            else:
                # decode element corresponding to node
                first_appearance = False
                dec = current_node.data
                code.append(current_node.data)
            self.update(dec, first_appearance)

        return code

    def update(self, data, first_appearance):

        def find_node_data(data):
            for node in self.all_nodes:
                if node.data == data:
                    return node
            raise KeyError(f'Cannot find the target node given {data}.')

        current_node = None
        while True:
            if first_appearance:
                current_node = self.nyt

                self.current_node_num -= 1
                new_external = Tree(1, self.current_node_num, data=data)
                current_node.right = new_external
                self.all_nodes.append(new_external)

                self.current_node_num -= 1
                self.nyt = Tree(0, self.current_node_num, data=NYT)
                current_node.left = self.nyt
                self.all_nodes.append(self.nyt)

                current_node.weight += 1
                current_node.data = None
                self.nyt = current_node.left
            else:
                if not current_node:
                    # First time as `current_node` is None.
                    current_node = find_node_data(data)
                node_max_num_in_block = max(
                    (
                        n for n in self.all_nodes
                        if n.weight == current_node.weight
                    ),
                    key=operator.attrgetter('num')
                )
                if node_max_num_in_block not in (current_node, current_node.parent):
                    exchange(current_node, node_max_num_in_block)
                    current_node = node_max_num_in_block
                current_node.weight += 1
            if not current_node.parent:
                break
            current_node = current_node.parent
            first_appearance = False

        self.symbols_count += 1
        if self.symbols_count >= self.INERTION:
            # print("halving weights")
            self.halve_weights()
            self.symbols_count = 0

    def halve_weights(self):
        # divide weights by half
        for node in self.all_nodes:
            node.weight = max(1, node.weight // 2)  #make sure weights dont drop under 1


def compress_text(text, alphabet_range=(0, 255), inertion=0):
    # Convert text to a byte sequence
    byte_seq = text.encode('utf-8')
    ada_huff = AdaptiveHuffman(byte_seq, alphabet_range, inertion)
    code = ada_huff.encode()
    return code


def extract_text(code, alphabet_range=(0, 255)):
    ada_huff = AdaptiveHuffman(None, alphabet_range)
    ada_huff.byte_seq = code.tobytes()
    decoded_bytes = ada_huff.decode()
    # Convert byte sequence back to text
    return bytes(decoded_bytes).decode('utf-8')


def read_text_from_out(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, 'out')
    file_path = os.path.join(out_dir, file_name)

    with open(file_path, 'r') as file:
        text_content = file.read()
    return text_content

def test_inertion(file_name, inertion_arr):
    text_string = read_text_from_out(file_name)


    results = []
    for i in reversed(inertion_arr):
        try:
            print(f"encoding with inertion {i}")
            encoded_bits = compress_text(text_string, inertion=i)
            compressed_size_in_bits = len(encoded_bits)
            compressed_size_in_bytes = compressed_size_in_bits / 8
            original_size_in_bytes = len(text_string.encode('utf-8'))
            percent_compression = ((original_size_in_bytes - compressed_size_in_bytes) / original_size_in_bytes) * 100

            # print(text_string)
            # print(encoded_bits)

            print(
                f"INERTION: {i}\t\t Compressed size: {compressed_size_in_bytes:.2f} bytes\t\tPercent Compression: {percent_compression:.10f}%")
            results.append([i, compressed_size_in_bytes, original_size_in_bytes, percent_compression])
        except Exception as e:
            print(f"Error at INERTION {i}: {e}")
            results.append([i, 'error', 'error', 'error'])


    with open(f"out/compression_results.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Inertion', 'Compressed size in bytes', 'Original Size in bytes', 'Percent of compression (%)'])
        writer.writerows(results)

    print("Results saved to out/'compression_results.csv'")



def run_compression(inertion, file_name):
    text_string = read_text_from_out(file_name)
    encoded_text = compress_text(text_string, inertion=inertion)
    with open(f"out/output_{inertion}.txt", "w") as text_file:
        text_file.write(encoded_text.to01())
    with open(f"out/output_{inertion}.bin", "wb") as binary_file:
        encoded_text.tofile(binary_file)
#
# def test_file():
#     file_name = 'short-text.txt'
#     text_string = read_text_from_out(file_name)
#
#     print(len(text_string))
#
#     encoded_text = compress_text(text_string)
#     print(len(encoded_text))
#
#     decoded_text = extract_text(encoded_text)
#     print(len(decoded_text))
#
#
# def test_inertion_for_values():
#     file_name = 'short-text.txt'
#     text_string = read_text_from_out(file_name)
#     print("reading")
#     for i in range(10, len(text_string)):
#         try:
#             encoded_text = compress_text(text_string, inertion=i)
#             print(f"INERTION: {i}\t\t Total compressed {len(encoded_text)}")
#         except:
#             print(f"error at INERTION {i}")
#

# test_inertion_for_values()


text_file_name = "initial-text.txt"

# inertions = [2**i for i in range(10, 24)]
# print(inertions)
# test_inertion(text_file_name, inertions)

appropriate_inertion = 1024
run_compression(appropriate_inertion, text_file_name)

