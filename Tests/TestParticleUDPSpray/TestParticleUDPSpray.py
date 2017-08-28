import socket
import time
import threading

UDP_IP = "192.168.0.120"
UDP_PORT = 7191
spray_off_msg = b"0-Off"
spray_on_msg = b"1-On"
lighting_msg = "L="
thread_run = True

def listen_for_udp(sock):
    global thread_run
    sock.connect((UDP_IP, UDP_PORT))
    sock.settimeout(2)
    while thread_run:
        try:
            data, addr = sock.recvfrom(1024)
            print("received message:", data, addr)
        except socket.timeout:
            pass

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
listen_UDP = threading.Thread(target=listen_for_udp, args=(sock,))
listen_UDP.start()
for i in range(2):

    if i % 2 == 0:
        sock.sendto(spray_off_msg, (UDP_IP, UDP_PORT))
        print("UDP {}:{} {}".format(UDP_IP, UDP_PORT, spray_off_msg))
    else:
        sock.sendto(spray_on_msg, (UDP_IP, UDP_PORT))
        print("UDP {}:{} {}".format(UDP_IP, UDP_PORT, spray_on_msg))

    light_lev_msg = bytes(lighting_msg + str(i*10), 'utf-8')
    sock.sendto(light_lev_msg, (UDP_IP, UDP_PORT))
    print("UDP {}:{} {}".format(UDP_IP, UDP_PORT, light_lev_msg))

    time.sleep(5.0)

thread_run = False
time.sleep(5.0)
