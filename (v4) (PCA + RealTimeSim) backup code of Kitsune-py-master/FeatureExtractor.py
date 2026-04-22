# Check if cython code has been compiled
import os
import subprocess
import time  # Import time module for sleep and timing
import shutil  # Import shutil for shutil.which
import logging # Added for logging warnings and errors
import csv # Ensure csv module is imported

# Set up logging for FeatureExtractor
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

use_extrapolation = False # experimental correlation code
if use_extrapolation:
    logging.info("Importing AfterImage Cython Library")
    if not os.path.isfile("AfterImage.c"): # has not yet been compiled, so try to do so...
        cmd = "python setup.py build_ext --inplace"
        subprocess.call(cmd, shell=True)
# Import dependencies
import netStat as ns
import numpy as np
logging.info("Importing Scapy Library")
from scapy.all import *
import os.path
import platform
import subprocess


# Extracts Kitsune features from given pcap file one packet at a time using "get_next_vector()"
# If wireshark is installed (tshark) it is used to parse (it's faster), otherwise, scapy is used (much slower).
# If wireshark is used then a tsv file (parsed version of the pcap) will be made -which you can use as your input next time
class FE:
    def __init__(self, file_path, limit=np.inf, live_stream=False, delimiter='\t'):
        # Fix: Ensure limit is always a numerical type
        self.limit = limit if limit is not None else np.inf 
        
        self.path = file_path
        self.parse_type = None  # unknown
        self.curPacketIndx = 0
        self.tsvin = None  # used for parsing TSV file
        self.scapyin = None  # used for parsing pcap with scapy
        self.last_input = None  # Store last processed feature vector
        self.last_timestamp = None  # Store last packet timestamp
        self.live_stream = live_stream  # Flag for live stream processing
        # --- REVERTED: Using passed 'delimiter' parameter ---
        self.input_delimiter = delimiter # <-- REVERTED TO USE PASSED PARAMETER
        # --- END REVERT ---
        self.live_stream_timeout = 5  # Timeout in seconds for live stream when no new data found
        self.live_stream_check_interval = 0.1 # Interval for checking for new data during live stream
        self._tshark = self.__locate_tshark()  # Ensure _tshark is initialized


        ### Prep pcap ##
        self.__prep__()

        self.nstat = ns.netStat()  # Initialize network statistics tracker
        self.num_features = len(self.nstat.getNetStatHeaders())  # Get initial feature count
        
        # Open TSV/CSV file if in live stream mode, to keep the file handle open
        if self.live_stream and (self.parse_type == "tsv" or self.parse_type == "csv"):
            try:
                self.tsvin = open(self.path, 'r', newline='') # Use newline='' for csv.reader compatibility
                # Attempt to skip header. If file is empty, readline might return empty string.
                header_line = self.tsvin.readline()
                if not header_line and os.path.getsize(self.path) > 0:
                    logging.warning(f"File '{self.path}' appears to have no header or is empty on startup.")
                # The delimiter printed here will now reflect the passed value from Kitsune.
                logging.info(f"Opened {self.path} for live TSV/CSV streaming with delimiter '{self.input_delimiter}'.")
            except Exception as e:
                logging.error(f"Error opening live stream file {self.path}: {e}")
                self.live_stream = False # Disable live stream if file can't be opened

    def __locate_tshark(self):
        # Determine tshark path based on OS
        if platform.system() == "Windows":
            # Common Wireshark installation paths
            wireshark_paths = [
                os.environ.get('WIRESHARK_BASE_DIR'),
                os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), 'Wireshark'),
                os.path.join(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'), 'Wireshark')
            ]
            for p in wireshark_paths:
                if p and os.path.exists(os.path.join(p, 'tshark.exe')):
                    return os.path.join(p, 'tshark.exe')
            # Fallback to system PATH
            if shutil.which("tshark"):
                return "tshark"
            logging.error("TShark not found. Please install Wireshark and ensure tshark is in your PATH or WIRESARK_BASE_DIR env variable is set.")
            return None
        else: # Linux, macOS
            if shutil.which("tshark"):
                return "tshark"
            logging.error("TShark not found. Please install Wireshark (apt-get install wireshark) or ensure tshark is in your PATH.")
            return None


    def __prep__(self):
        # Determine parse type based on file extension
        if self.path.endswith(".pcap"):
            self.parse_type = "pcap"
            # Scapy's PcapReader for non-tshark parsing
            try:
                self.scapyin = PcapReader(self.path)
            except Exception as e:
                logging.error(f"Error opening pcap with PcapReader: {e}")
                self.scapyin = None
                self.parse_type = None
        elif self.path.endswith(".tsv") or self.path.endswith(".csv"):
            self.parse_type = "tsv" # Treat .csv as tsv for generic parsing via csv.reader
        else:
            logging.error("Invalid file type. Must be .pcap, .tsv, or .csv")
            sys.exit(1)


    def get_num_features(self):
        """Returns the number of features generated by the feature extractor."""
        return self.num_features

    def get_latest_timestamp(self):
        """Returns the timestamp of the last processed packet."""
        return self.last_timestamp


    def get_next_vector(self):
        """
        Extracts features from the next packet in the stream.
        Returns a 1D numpy array of features or an empty list if EOF or error.
        """
        if self.curPacketIndx >= self.limit:
            return []

        if self.parse_type == "pcap":
            if self.scapyin is None:
                logging.warning("Scapy PcapReader not initialized. Cannot read PCAP.")
                return [] # No scapy reader available

            packet = self.scapyin.read_packet()
            if packet is None: # EOF for pcap
                return []
            self.last_timestamp = float(packet.time)

            # Extract common network features using scapy
            srcIP = ''
            dstIP = ''
            srcMAC = ''
            dstMAC = ''
            srcproto = ''
            dstproto = ''
            framelen = len(packet)

            if 'Ether' in packet:
                srcMAC = packet['Ether'].src
                dstMAC = packet['Ether'].dst

            if 'IP' in packet:
                srcIP = packet['IP'].src
                dstIP = packet['IP'].dst
                # Check for protocol specific layers for ports
                if 'TCP' in packet:
                    srcproto = str(packet['TCP'].sport)
                    dstproto = str(packet['TCP'].dport)
                elif 'UDP' in packet:
                    srcproto = str(packet['UDP'].sport)
                    dstproto = str(packet['UDP'].dport)
                elif 'ICMP' in packet:
                    srcproto = 'icmp'
                    dstproto = 'icmp'
            elif 'ARP' in packet:
                srcIP = packet['ARP'].psrc # source IP
                dstIP = packet['ARP'].pdst # destination IP
                srcproto = 'arp'
                dstproto = 'arp'
            elif 'IPv6' in packet: # Handle IPv6
                srcIP = packet['IPv6'].src
                dstIP = packet['IPv6'].dst
                if 'TCP' in packet:
                    srcproto = str(packet['TCP'].sport)
                    dstproto = str(packet['TCP'].dport)
                elif 'UDP' in packet:
                    srcproto = str(packet['UDP'].sport)
                    dstproto = str(packet['UDP'].dport)
                elif 'ICMPv6' in packet:
                    srcproto = 'icmpv6'
                    dstproto = 'icmpv6'
            else: # Some other protocol
                # For non-IP/ARP, try to extract MACs if Ethernet is present
                if 'Ether' in packet:
                    srcIP = packet['Ether'].src # src MAC as IP for feature extraction logic
                    dstIP = packet['Ether'].dst # dst MAC as IP for feature extraction logic
                else:
                    return [] # Cannot extract meaningful features for unknown layer 2 or higher

        elif self.parse_type == "tsv":
            if self.tsvin is None: # Should be open for live stream, or handled by pcap2tsv_with_tshark otherwise
                logging.warning("TSV/CSV input stream not initialized. Cannot read TSV/CSV.")
                return []

            line = ""
            if self.live_stream:
                # Loop to wait for new data if EOF is hit
                start_wait_time = time.time()
                while not line:
                    current_pos = self.tsvin.tell()
                    self.tsvin.seek(0, os.SEEK_END)
                    end_pos = self.tsvin.tell()
                    self.tsvin.seek(current_pos) # Reset file pointer

                    if current_pos == end_pos: # If at EOF
                        if (time.time() - start_wait_time) > self.live_stream_timeout:
                            logging.info(f"Live stream timeout ({self.live_stream_timeout}s) reached. No new data.")
                            return [] # Timeout reached, signal end of stream
                        time.sleep(self.live_stream_check_interval) # Small sleep to avoid busy-waiting
                    else: # New data is available
                        line = self.tsvin.readline()
                        if not line: # Should not happen if current_pos != end_pos, but safety check
                            continue
            else: # Not live stream, just read once
                line = self.tsvin.readline()
                if not line: # EOF for non-live stream
                    return []
            
            # Use csv.reader for robust parsing of the TSV/CSV line
            try:
                reader = csv.reader([line.strip()], delimiter=self.input_delimiter) 
                row = next(reader)
            except csv.Error as e:
                logging.warning(f"CSV parsing error on line: '{line.strip()}'. Error: {e}. Skipping row.")
                return []

            # Ensure row has enough columns before accessing them
            # Based on the ciniminer_sample_traffic_log.csv structure for common fields
            expected_min_cols = 8 # timestamp, protocol, srcIP, srcPort, srcMAC, dstIP, dstPort, dstMAC, framelen (indices 0 to 8)
            if len(row) < expected_min_cols:
                logging.warning(f"Skipping malformed row (too few columns, expected at least {expected_min_cols}): {line.strip()}")
                return []

            try:
                # Robust conversion and NaN/empty string checks for critical fields
                current_timestamp_str = row[0].strip()
                if not current_timestamp_str or current_timestamp_str.lower() == 'nan':
                    logging.warning(f"Skipping row {self.curPacketIndx + 1}: Invalid timestamp '{current_timestamp_str}'. Line: {line.strip()}")
                    return []
                self.last_timestamp = float(current_timestamp_str)

                framelen_str = row[8].strip() if len(row) > 8 else ''
                if not framelen_str or framelen_str.lower() == 'nan':
                    logging.warning(f"Skipping row {self.curPacketIndx + 1}: Invalid frame length '{framelen_str}'. Line: {line.strip()}")
                    return []
                framelen = int(framelen_str)

                srcMAC = row[4].strip() if len(row) > 4 else ''
                dstMAC = row[7].strip() if len(row) > 7 else ''
                srcIP = row[2].strip() if len(row) > 2 else ''
                dstIP = row[5].strip() if len(row) > 5 else ''
                
                protocol = row[1].strip() if len(row) > 1 else ''
                srcproto = row[3].strip() if len(row) > 3 else ''
                dstproto = row[6].strip() if len(row) > 6 else ''

                # Basic validation for essential network fields
                if not (srcIP and dstIP and srcMAC and dstMAC):
                    logging.warning(f"Skipping row {self.curPacketIndx + 1}: Missing essential network fields (IP/MAC). Line: {line.strip()}")
                    return []

                # Handle specific protocols if needed by netStat
                if protocol.lower() == 'icmp':
                    srcproto = 'icmp'
                    dstproto = 'icmp'
                elif protocol.lower() == 'arp':
                    srcproto = 'arp'
                    dstproto = 'arp'
                
            except (ValueError, IndexError, TypeError) as e:
                logging.warning(f"Error parsing row data at index {self.curPacketIndx + 1}: {e} - Line: '{line.strip()}'. Skipping row.")
                return [] # Skip this row if parsing fails
        else:
            logging.error(f"Unknown parse type: {self.parse_type}. Returning empty vector.")
            return [] # Unknown parse type

        self.curPacketIndx = self.curPacketIndx + 1

        ### Extract Features
        try:
            # Determine IP type (IPv4, IPv6, or non-IP for MAC-based)
            IPtype = -1 # Unknown
            if ':' in srcIP and ':' in dstIP:
                IPtype = 1 # IPv6
            elif '.' in srcIP and '.' in dstIP:
                IPtype = 0 # IPv4
            # If not IP, netStat expects MACs to be passed as IPs, so IPtype remains -1 (or can be adjusted in netStat if needed)

            self.last_input = self.nstat.updateGetStats(IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto,
                                                 int(framelen),
                                                 self.last_timestamp)
            return self.last_input
        except Exception as e:
            logging.warning(f"Error extracting features from packet {self.curPacketIndx}: {e}. Skipping.")
            return []


    def get_next_batch_vectors(self, batch_size):
        """
        Collects a batch of feature vectors.
        Args:
            batch_size (int): The number of vectors to attempt to collect.
        Returns:
            Tuple[np.ndarray, list]: A tuple containing:
                - A 2D NumPy array of feature vectors (n_samples, n_features).
                - A list of timestamps for each feature vector.
        """
        feature_vectors = []
        timestamps = []
        for _ in range(batch_size):
            vector = self.get_next_vector()
            
            if not isinstance(vector, (list, np.ndarray)):
                logging.warning(f"Unexpected type received from get_next_vector: {type(vector)}. Skipping this unexpected vector.")
                continue # Skip this unexpected type and try to get the next one

            # If it's a NumPy array, check its size; if it's a list, check its length (for emptiness)
            if (isinstance(vector, np.ndarray) and vector.size == 0) or \
               (isinstance(vector, list) and len(vector) == 0):
                # This means it's an empty array or empty list, signaling EOF or an error in get_next_vector
                break # Stop collecting batch if no more valid vectors are available

            feature_vectors.append(vector)
            timestamps.append(self.get_latest_timestamp())

        if not feature_vectors:
            return np.array([]), [] # Return empty arrays if no vectors collected
        
        num_features = self.get_num_features()
        padded_vectors = []
        for vec in feature_vectors:
            if len(vec) == num_features:
                padded_vectors.append(vec)
            else:
                # Handle cases where a vector might have fewer features than expected
                # This can happen if some parsing fails for specific fields in a row.
                logging.warning(f"Padding feature vector from {len(vec)} to {num_features} features.")
                padded_vec = np.zeros(num_features)
                padded_vec[:len(vec)] = vec[:num_features]
                padded_vectors.append(padded_vec)

        if not padded_vectors:
            return np.array([]), []

        return np.array(padded_vectors), timestamps


    def pcap2tsv_with_tshark(self):
        """
        Converts a pcap file to a TSV file using tshark.
        """
        if not self._tshark:
            logging.error("TShark executable not found. Cannot convert pcap to tsv.")
            return

        logging.info('Parsing with tshark...')
        # Note: The original fields covered common network protocols.
        # Ensure these fields are relevant for your specific log analysis.
        fields = "-e frame.time_epoch -e frame.len -e eth.src -e eth.dst -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -e icmp.type -e icmp.code -e arp.opcode -e arp.src.hw_mac -e arp.src.proto_ipv4 -e arp.dst.hw_mac -e arp.dst.proto_ipv4 -e ipv6.src -e ipv6.dst"
        cmd = (
            f'"{self._tshark}" -r "{self.path}" -T fields {fields} '
            f'-E header=y -E occurrence=f > "{self.path}.tsv"'
        )
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            logging.info(f"tshark parsing complete. File saved as: {self.path}.tsv")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error during tshark conversion: {e.stderr}")
        except FileNotFoundError:
            logging.error(f"TShark command not found. Please ensure '{self._tshark}' is correctly installed and accessible.")

