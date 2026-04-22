#Check if cython code has been compiled
import os
import subprocess
import sys # Added for sys.maxsize

use_extrapolation=False #experimental correlation code
if use_extrapolation:
    print("Importing AfterImage Cython Library")
    if not os.path.isfile("AfterImage.c"): #has not yet been compiled, so try to do so...
        cmd = "python setup.py build_ext --inplace"
        subprocess.call(cmd,shell=True)
#Import dependencies
import netStat as ns
import csv
import numpy as np
# import cupy as np
print("Importing Scapy Library")
from scapy.all import *
import os.path
import platform
import subprocess
import time # Added for sleeping in live stream mode


#Extracts Kitsune features from given pcap file one packet at a time using "get_next_vector()"
# If wireshark is installed (tshark) it is used to parse (it's faster), otherwise, scapy is used (much slower).
# If wireshark is used then a tsv file (parsed version of the pcap) will be made -which you can use as your input next time
class FE:
    def __init__(self,file_path,limit=np.inf, live_stream=False, delimiter='\t'): # Added live_stream and delimiter parameters
        self.path = file_path
        self.limit = limit
        self.parse_type = None #unknown
        self.curPacketIndx = 0
        self.tsvin = None #used for parsing TSV file
        self.tsvinf = None #file handle for tsv
        self.scapyin = None #used for parsing pcap with scapy
        self.last_input = None  # Store last processed feature vector
        self.last_timestamp = None  # Store last packet timestamp
        self.live_stream = live_stream # New parameter to indicate live streaming
        self.delimiter = delimiter # Store the delimiter

        ### Prep pcap ##
        self.__prep__()

        ### Prep Feature extractor (AfterImage) ###
        maxHost = 100000000000
        maxSess = 100000000000
        self.nstat = ns.netStat(np.nan, maxHost, maxSess)

    def _get_tshark_path(self):
        if platform.system() == 'Windows':
            return 'C:/Program Files/Wireshark/tshark.exe'
        else:
            system_path = os.environ['PATH']
            for path in system_path.split(os.pathsep):
                filename = os.path.join(path, 'tshark')
                if os.path.isfile(filename):
                    return filename
        return ''

    def __prep__(self):
        ### Find file: ###
        if not os.path.isfile(self.path):  # file does not exist
            print("File: " + self.path + " does not exist. Waiting for file to appear...")
            # In live stream mode, we might wait for the file to be created
            if self.live_stream:
                timeout = 60 # seconds
                start_time = time.time()
                while not os.path.isfile(self.path):
                    if time.time() - start_time > timeout:
                        print(f"Timeout: File {self.path} did not appear within {timeout} seconds.")
                        raise Exception(f"File {self.path} not found and timed out waiting.")
                    time.sleep(1) # Wait for file to be created
                print(f"File {self.path} found.")
            else:
                print("File: " + self.path + " does not exist")
                raise Exception()

        ### check file type ###
        type = self.path.split('.')[-1]

        # Check if tshark is available (only relevant for .pcap files)
        self._tshark = self._get_tshark_path()

        ##If file is TSV/CSV
        if type == "tsv" or type == "csv": # Allow .csv extension for TSV parsing
            self.parse_type = "tsv" # Treat as TSV for parsing purposes

        ##If file is pcap
        elif type == "pcap" or type == 'pcapng':
            # Try parsing via tshark dll of wireshark (faster)
            # We only use tshark conversion for static pcap files, not live streams
            if os.path.isfile(self._tshark) and not self.live_stream:
                self.pcap2tsv_with_tshark()  # creates local tsv file
                self.path += ".tsv" # Update path to the newly created tsv
                self.parse_type = "tsv"
            else: # Otherwise, parse with scapy (slower) or if live_stream is true
                if self.live_stream:
                    print("Live stream mode: PCAP input is not directly supported for live streaming. Please provide a TSV/CSV file.")
                    raise Exception("Live stream mode requires TSV/CSV input.")
                print("tshark not found or live stream mode. Trying scapy...")
                self.parse_type = "scapy"
        else:
            print("File: " + self.path + " is not a tsv, csv, or pcap file")
            raise Exception()

        ### open readers ##
        if self.parse_type == "tsv":
            maxInt = sys.maxsize
            decrement = True
            while decrement:
                decrement = False
                try:
                    csv.field_size_limit(maxInt)
                except OverflowError:
                    maxInt = int(maxInt / 10)
                    decrement = True

            if self.live_stream:
                # In live stream mode, open in read mode and seek to end initially
                # We will continuously read new lines as they are appended
                self.tsvinf = open(self.path, 'rt', encoding="utf8")
                self.tsvinf.seek(0, os.SEEK_END) # Go to end of file
                self.tsvin = csv.reader(self.tsvinf, delimiter=self.delimiter) # Use specified delimiter
                print(f"Opened {self.path} for live TSV/CSV streaming with delimiter '{self.delimiter}'.")
            else:
                print("counting lines in file...")
                num_lines = sum(1 for line in open(self.path, 'rt', encoding="utf8")) # Count lines with utf8 encoding
                print("There are " + str(num_lines) + " Packets.")
                self.limit = min(self.limit, num_lines-1) # -1 for header
                self.tsvinf = open(self.path, 'rt', encoding="utf8")
                self.tsvin = csv.reader(self.tsvinf, delimiter=self.delimiter) # Use specified delimiter
                try:
                    row = self.tsvin.__next__() #move iterator past header
                    print(f"Skipped header: {row}") # Diagnostic print
                except StopIteration:
                    print("TSV/CSV file is empty or only has a header.")
                    self.limit = 0 # No data to process
                    return # Exit __prep__ early for empty files

        else: # scapy
            print("Reading PCAP file via Scapy...")
            self.scapyin = rdpcap(self.path)
            self.limit = len(self.scapyin)
            print("Loaded " + str(len(self.scapyin)) + " Packets.")

    def get_next_vector(self):
        if self.parse_type == "tsv":
            row = None # Initialize row to None
            while True: # Loop to wait for new data in live stream mode
                try:
                    if not self.live_stream and self.curPacketIndx >= self.limit:
                        self.tsvinf.close()
                        return [] # End of static file

                    row = self.tsvin.__next__()
                    self.curPacketIndx += 1 # Increment only when a row is successfully read

                    # --- Column Mapping for conn.log format ---
                    # Based on: ts|uid|id.orig_h|id.orig_p|id.resp_h|id.resp_p|proto|service|duration|orig_bytes|resp_bytes|...

                    # 0: ts (timestamp)
                    self.last_timestamp = 0.0
                    try:
                        self.last_timestamp = float(row[0])
                    except (ValueError, IndexError):
                        print(f"Warning: Could not parse timestamp from row[0]: '{row[0]}'. Using 0.0.")

                    # 9: orig_bytes, 10: resp_bytes (using for framelen)
                    framelen = 60 # Default frame length
                    try:
                        orig_bytes = float(row[9]) if len(row) > 9 and row[9] != '-' else 0.0
                        resp_bytes = float(row[10]) if len(row) > 10 and row[10] != '-' else 0.0
                        # Use total bytes as framelen proxy, ensure it's an int
                        framelen = int(orig_bytes + resp_bytes)
                        if framelen == 0: # Ensure a minimum frame length if sum is zero
                            framelen = 60
                    except (ValueError, IndexError):
                        print(f"Warning: Could not parse framelen from row[9]/row[10]. Using default {framelen}.")
                        # framelen remains 60

                    # 2: id.orig_h (source IP)
                    srcIP = row[2] if len(row) > 2 and row[2] != '-' else ''
                    # 4: id.resp_h (destination IP)
                    dstIP = row[4] if len(row) > 4 and row[4] != '-' else ''

                    IPtype = np.nan # 0 for IPv4, 1 for IPv6
                    if ':' in srcIP or ':' in dstIP: # Simple check for IPv6
                        IPtype = 1
                    elif '.' in srcIP or '.' in dstIP: # Simple check for IPv4
                        IPtype = 0

                    # 6: proto (protocol name: tcp, udp, icmp, etc.)
                    protocol_name = row[6].lower() if len(row) > 6 else ''

                    # 3: id.orig_p (source port)
                    srcport = row[3] if len(row) > 3 and row[3] != '-' else ''
                    # 5: id.resp_p (destination port)
                    dstport = row[5] if len(row) > 5 and row[5] != '-' else ''

                    # Construct srcproto and dstproto for netStat
                    # netStat expects protocol (e.g., 'tcp') + port (e.g., '80')
                    srcproto = f"{protocol_name}{srcport}" if srcport else protocol_name
                    dstproto = f"{protocol_name}{dstport}" if dstport else protocol_name

                    # MAC addresses are not in conn.log, use IPs as host identifiers for netStat
                    srcMAC = srcIP
                    dstMAC = dstIP

                    # Handle special protocols
                    if protocol_name == 'arp':
                        srcproto = 'arp'
                        dstproto = 'arp'
                        # For ARP, netStat uses arp.src.proto_ipv4 and arp.dst.proto_ipv4 as IPs
                        # Your data has id.orig_h and id.resp_h as the IPs, which is fine.
                    elif protocol_name == 'icmp':
                        srcproto = 'icmp'
                        dstproto = 'icmp'
                    elif not srcIP and not dstIP and not srcMAC and not dstMAC: # Fallback for truly unknown L1/L2
                        srcMAC = 'unknown_host_src'
                        dstMAC = 'unknown_host_dst'


                    break # Successfully read a row, break from while True loop

                except StopIteration:
                    if self.live_stream:
                        # No new data yet, wait and try again
                        time.sleep(0.1) # Small sleep to avoid busy-waiting
                        # Re-create csv.reader to pick up new lines
                        self.tsvinf.seek(self.tsvinf.tell()) # Seek to current position to clear EOF flag
                        self.tsvin = csv.reader(self.tsvinf, delimiter=self.delimiter)
                        continue # Try reading again
                    else:
                        print(f"End of file reached for {self.path}")
                        return [] # End of static file
                except Exception as e:
                    print(f"Error parsing TSV/CSV row at index {self.curPacketIndx}: {e}. Row data: {row}")
                    # If an error occurs, try to read the next line to avoid getting stuck
                    # Or return empty if data is consistently malformed
                    return [] # Return empty on error for this packet

        elif self.parse_type == "scapy":
            if self.curPacketIndx >= self.limit: # Changed to >=
                return []

            packet = self.scapyin[self.curPacketIndx]
            IPtype = np.nan
            self.last_timestamp = float(packet.time)  # Store timestamp
            timestamp = packet.time
            framelen = len(packet)
            if packet.haslayer(IP):  # IPv4
                srcIP = packet[IP].src
                dstIP = packet[IP].dst
                IPtype = 0
            elif packet.haslayer(IPv6):  # ipv6
                srcIP = packet[IPv6].src
                dstIP = packet[IPv6].dst
                IPtype = 1
            else:
                srcIP = ''
                dstIP = ''

            if packet.haslayer(TCP):
                srcproto = str(packet[TCP].sport)
                dstproto = str(packet[TCP].dport)
            elif packet.haslayer(UDP):
                srcproto = str(packet[UDP].sport)
                dstproto = str(packet[UDP].dport)
            else:
                srcproto = ''
                dstproto = ''

            srcMAC = packet.src
            dstMAC = packet.dst
            if srcproto == '':  # it's a L2/L1 level protocol
                if packet.haslayer(ARP):  # is ARP
                    srcproto = 'arp'
                    dstproto = 'arp'
                    srcIP = packet[ARP].psrc  # src IP (ARP)
                    dstIP = packet[ARP].pdst  # dst IP (ARP)
                    IPtype = 0
                elif packet.haslayer(ICMP):  # is ICMP
                    srcproto = 'icmp'
                    dstproto = 'icmp'
                    IPtype = 0
                elif srcIP + srcproto + dstIP + dstproto == '':  # some other protocol
                    srcIP = packet.src  # src MAC
                    dstIP = packet.dst  # dst MAC
            self.curPacketIndx = self.curPacketIndx + 1 # Increment for scapy as well

        else: # Should not happen with proper parse_type
            return []

        ### Extract Features
        try:
            # Ensure all protocol/IP/MAC fields are strings before passing to nstat
            srcIP_str = str(srcIP)
            dstIP_str = str(dstIP)
            srcMAC_str = str(srcMAC)
            dstMAC_str = str(dstMAC)
            srcproto_str = str(srcproto)
            dstproto_str = str(dstproto)

            self.last_input = self.nstat.updateGetStats(IPtype, srcMAC_str, dstMAC_str, srcIP_str, srcproto_str, dstIP_str, dstproto_str,
                                                 int(framelen), # framelen is already int or default
                                                 self.last_timestamp) # Use stored timestamp
            return self.last_input  # Return and store last input
        except Exception as e:
            print(f"Error extracting features for packet {self.curPacketIndx}: {e}")
            return []


    def pcap2tsv_with_tshark(self):
        print('Parsing with tshark...')
        fields = "-e frame.time_epoch -e frame.len -e eth.src -e eth.dst -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -e icmp.type -e icmp.code -e arp.opcode -e arp.src.hw_mac -e arp.src.proto_ipv4 -e arp.dst.hw_mac -e arp.dst.proto_ipv4 -e ipv6.src -e ipv6.dst"
        cmd =  '"' + self._tshark + '" -r '+ self.path +' -T fields '+ fields +' -E header=y -E occurrence=f > '+self.path+".tsv"
        subprocess.call(cmd,shell=True)
        print("tshark parsing complete. File saved as: "+self.path +".tsv")

    def get_num_features(self):
        """
        Returns the number of features that netStat will produce.
        This is typically called after netStat is initialized in __init__.
        """
        return len(self.nstat.getNetStatHeaders())

    def get_latest_timestamp(self):
        """Get the timestamp of the last processed packet"""
        return self.last_timestamp

