from datetime import datetime, timedelta
from pathlib import Path

next_save_time = datetime.now() + timedelta(seconds=15)
output = Path("C:/Users/admin/Desktop/Project_VSC/Data/captured_packets.txt")
writer = open(output, "w", newline="")
print("open1")
writer.write("Time, Source IP, Destination IP")

while True:
    if datetime.now() <= next_save_time:
        # Save the captured packets to a CSV file
        pass
    else:

        writer.close()
        
        # Reopen the CSV writer with a new file
        output = Path("C:/Users/admin/Desktop/Project_VSC/Data/captured_packets.txt")
        writer = open(output, "w", newline="")
        print("open")
        writer.write(["Time", "Source IP", "Destination IP", "Source Port", "Destination Port", "TCP Flag", "Protocol", "Packets", "Label"])
        
        # Calculate the next save time
        next_save_time = datetime.now() + timedelta(seconds=15)

    # Check if the user pressed Enter to stop capturing
    #if input("Press Enter to stop capturing...") == "":
        #break