# Echo server program
import pickle
import socket

from core.utils import Msg
import sys
# hard code hosts' (VMs') ip and port here
# todo: use config file instead
HOST = socket.gethostname()
PORT = 55558

class Server:
    def __init__(self, host=HOST, port=PORT):
        """
        Server initialization.
        Make sure the server has already known the .log file.
        :param host: server host
        :param port: server post
        """
        self.host = host
        self.port = port

    def run(self):
        """
        Run a server, to receive the query pattern from clients and return log data.
        :return: None
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen(1)
            while True:
                print('[INFO]: Waiting for connection ...')
                conn, addr = s.accept()
                with conn:
                    print('[INFO]: Connected by', addr, ' on ', socket.gethostname())
                    try:
                        data = b""
                        while True:
                            temp = conn.recv(4096)
                            print(type(temp))
                            if temp:
                                print('[INFO] Received {} bytes'.format(len(temp)))
                                data += temp
                                print('[INFO] Total data {} bytes'.format(len(data)))
                            else:
                                print('[INFO] All arguments received, start parsing request ...')
                                break
                            print('INLOOP')
                        print('OUT !!!')
                        if data:
                            func_args = pickle.loads(data)
                            print('[INFO] Received {} search arguments for simulation'.format(len(func_args)))
                            data = pickle.dumps(Msg('DONE'))
                            conn.sendall(data)

                    except Exception as e:
                        print('[ERROR]:', e.__str__())


if __name__ == '__main__':
    s = Server()
    s.run()
