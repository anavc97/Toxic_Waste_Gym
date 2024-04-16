from http.server import SimpleHTTPRequestHandler, HTTPServer
import json
import threading
import time
import queue
import requests
import os
# Global variable for the current log ID
log_id = None

# Global variable for the path to the JSON file
json_file_handle = None
script_path = os.path.dirname(os.path.realpath(__file__))

class MyServer(SimpleHTTPRequestHandler):
    
    message_queue = queue.Queue()
    gameStarted = False

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = post_data.decode('utf-8')

        # Process received message and update internal variable
        respond = self.process_message(data)

        self._set_response()
        self.wfile.write(respond)

    def process_message(self, data):

        global log_id
        global json_file_handle

        print("Received message from Unity:", data)
        MyServer.message_queue.put(data)
        if isinstance(data, str) and data == "game ended":
            # Close the JSON file handle and save it locally
            if not json_file_handle.closed:
                json_file_handle.close()
            return b"Game ended. JSON file saved."

        elif isinstance(data, str) and data.isalnum():
            # If data is a single number with letters
            log_id = data
            json_file_path = os.path.join(script_path, f"logfile_{log_id}.json")
            print(json_file_path)
            # Close the existing file handle if it's open
            if json_file_handle:
                json_file_handle.close()
            # Open a new JSON file handle
            json_file_handle = open(json_file_path, 'a')
            return b"Log ID set. New JSON file opened."

        else:
            jsondata = json.loads(data)
            log_id = jsondata['id']
            print("writing log for id: ", log_id)
            json_file_path = os.path.join(script_path, f"logfile_{log_id}.json")
            json_file_handle = open(json_file_path, 'a')
            # Treat data as a line from a JSON file and append it
            if json_file_handle:
                json_file_handle.write(json.dumps(jsondata))
                json_file_handle.write('\n')  # Add a newline after each JSON object
                json_file_handle.close()
                return b"Data appended to JSON file."
            else:
                return b"No JSON file handle open. Data not saved."

def run_server():
    server_address = ('127.0.0.1', 5100)
    httpd = HTTPServer(server_address, MyServer)
    print('Starting server...')
    httpd.serve_forever()

def send_message_to_unity(message):
    url = "http://127.0.0.1:20500"  
    payload = {"message": message}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            print("Message sent to Unity successfully.")
        else:
            print("Failed to send message to Unity. Status code:", response.status_code)
    except Exception as e:
        print("An error occurred while sending message to Unity:", e)

def start_communication_loop():
    while True:
        # Send messages to Unity
        send_message_to_unity("Hello from Python!")
        # Do other tasks here
        time.sleep(5)  # Send message every 5 seconds

def main():
    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    #communication_thread = threading.Thread(target=start_communication_loop)
    #communication_thread.start()

if __name__ == '__main__':
	main()
