import pickle
from datetime import datetime
from multiprocessing import Pool
from socket import *
from struct import unpack

from tqdm import tqdm

from main_hyper_param_search import multi_run_wrapper


class ServerProtocol:
    def __init__(self):
        self.socket = None
        self.output_dir = '.'
        self.file_num = 1
        self.host = -1
        self.port = -1

    def listen(self, server_ip, server_port):
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.bind((server_ip, server_port))
        self.socket.listen(1)
        self.host = server_ip
        self.port = server_port

    def handle_data(self):
        try:
            while True:
                print('[INFO]: Waiting for connection ...')
                (connection, addr) = self.socket.accept()
                try:
                    bs = connection.recv(8)
                    (length,) = unpack('>Q', bs)
                    data = b''
                    while len(data) < length:
                        # doing it in batches is generally better than trying
                        # to do it all in one go, so I believe.
                        to_read = length - len(data)
                        data += connection.recv(
                            4096 if to_read > 4096 else to_read)

                    arguments = pickle.loads(data)
                    print('[INFO]: Received {} arguments for simulation'.format(len(arguments)))

                    with open("{}_receiver_test.txt".format(self.host), "wb") as fp:  # Pickling
                        pickle.dump(data, fp)

                    print('[INFO]: Starting simulations ...')

                    pool = Pool(processes=8)
                    starting = datetime.now()
                    print('staring at', starting)
                    for _ in tqdm(pool.map(multi_run_wrapper, arguments), total=len(arguments)):
                        pass
                    ending = datetime.now()
                    print('end at', ending)

                    # send our 0 ack
                    assert len(b'\00') == 1
                    connection.sendall(b'\00')
                finally:
                    connection.shutdown(SHUT_WR)
                    connection.close()

                self.file_num += 1
        finally:
            self.close()

    def close(self):
        self.socket.close()
        self.socket = None

        # could handle a bad ack here, but we'll assume it's fine.


if __name__ == '__main__':
    sp = ServerProtocol()
    sp.listen(gethostname(), 55558)
    sp.handle_data()
