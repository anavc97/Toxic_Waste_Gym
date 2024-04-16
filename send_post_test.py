import requests

def send_post_request(ip, port):
    url = f"http://{ip}:{port}/"
    data = {"key": "value"}  # Modify this dictionary with your POST data
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("POST request sent successfully!")
        else:
            print(f"Failed to send POST request. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace 'your_ip' and 'your_port' with the specific IP address and port you want to send the request to
ip = '146.193.224.2'
port = '5000'

send_post_request(ip, port)