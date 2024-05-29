hex_number = "0x00c2"
binary_number = bin(int(hex_number, 16))[2:]  # Convert hexadecimal to binary
binary_number = binary_number.zfill(8)  # Pad with zeros to ensure a 16-bit binary representation

if binary_number[3:4] == "1":
    print("ACK\n")
if binary_number[4:5] == "1":
    print("PSH\n")
if binary_number[5:6] == "1":
    print("RST\n")
if binary_number[6:7] == "1":
    print("SYN\n")
if binary_number[7:] == "1":
    print("FIN\n")

print(binary_number)
print(binary_number[0:1])
print(binary_number[1:2])
print(binary_number[2:3])
print(binary_number[3:4])
print(binary_number[4:5])
print(binary_number[5:6])
print(binary_number[6:7])
print(binary_number[7:])