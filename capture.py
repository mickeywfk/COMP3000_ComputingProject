from scapy.all import sniff
from datetime import datetime, timedelta
import csv
from pathlib import Path

# Set the capture duration
capture_duration = timedelta(seconds=60)
save_interval = timedelta(seconds=15)

def packet_handler(packet):
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(packet)
    if packet.haslayer("IP"):
        source_ip = packet.getlayer("IP").src
        destination_ip = packet.getlayer("IP").dst
    elif packet.haslayer("ARP"):
        source_ip = packet.getlayer("ARP").psrc
        destination_ip = packet.getlayer("ARP").pdst
        protocol = "ARP"
    else:
        source_ip = "Unknown"
        destination_ip = "Unknown"
        protocol = "Unknown"
    
    if packet.haslayer("IP"):
        if packet.haslayer("TCP"):
            protocol = "TCP"
        elif packet.haslayer("UDP"):
            protocol = "UDP"
        elif packet.haslayer("ICMP"):
            protocol = "ICMP"
        else:
            protocol = "Unknown"

    packet_size = len(packet)
    

    # Check if the packet has a TCP or UDP layer
    if packet.haslayer("TCP"):
        source_port = packet.getlayer("TCP").sport
        destination_port = packet.getlayer("TCP").dport
        tcp_flag = hex(int(packet.getlayer("TCP").flags))
    elif packet.haslayer("UDP"):
        source_port = packet.getlayer("UDP").sport
        destination_port = packet.getlayer("UDP").dport
        tcp_flag = 0
    elif packet.haslayer("ARP"):
        source_port = 0
        destination_port = 0
        tcp_flag = 0
    else:
        source_port = 0
        destination_port = 0
        tcp_flag = 0

    # Write packet information to the CSV file
    with open(output, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([timestamp, source_ip, destination_ip, source_port, destination_port, tcp_flag, protocol, packet_size])

# Calculate the end time for packet capture
end_time = datetime.now() + capture_duration

# Calculate the next save time
next_save_time = datetime.now() + save_interval

time = datetime.now().strftime("%d%m%Y%H%M%S")
output = f"C:/Users/admin/Desktop/Project_VSC/Data/captured_packets_{time}.csv"
with open(output, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    if csvfile.tell() == 0:  # Check if the file is empty
        writer.writerow(["Time", "Source IP", "Destination IP", "Source Port", "Destination Port", "TCP Flag", "Protocol", "Packets", "Label"])

# Start capturing packets
while datetime.now() < end_time:
    if datetime.now() >= next_save_time:
        # Save the captured packets to a CSV file
        sniff(prn=packet_handler, store=False, timeout=save_interval.total_seconds())

        # Calculate the next save time
        next_save_time = datetime.now() + save_interval
        time = datetime.now().strftime("%d%m%Y%H%M%S")
        output = f"C:/Users/admin/Desktop/Project_VSC/Data/captured_packets_{time}.csv"
        with open(output, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:  # Check if the file is empty
                writer.writerow(["Time", "Source IP", "Destination IP", "Source Port", "Destination Port", "TCP Flag", "Protocol", "Packets", "Label"])

# Inform that packet capture has ended
print("Packet capture ended.")