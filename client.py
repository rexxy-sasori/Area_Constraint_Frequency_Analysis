import pickle
import socket
import threading
import time
from socket import *
from struct import pack

from IO import config

HOSTS = [
    'csl-411-05.csl.illinois.edu',
    'csl-411-06.csl.illinois.edu',

    'csl-411-07.csl.illinois.edu',
    'csl-411-08.csl.illinois.edu',

    'csl-411-09.csl.illinois.edu',
    'csl-411-10.csl.illinois.edu',

    'csl-411-13.csl.illinois.edu',
    'csl-411-14.csl.illinois.edu',

    'aristotle.csl.illinois.edu'
]
PORT = 55558


class QueryThread(threading.Thread):
    def __init__(self, host, port):
        """
        Define thread for query.

        :param pattern: query string, e.g. 'a'(raw string), 'a[a-z]b'(regex)
        :param host: host of query target
        :param port: port of query target
        """
        super(QueryThread, self).__init__()
        self.func_args = []
        self.host = host
        self.port = port
        self.time_cost = -1.0  # record time cost for single thread

    def run(self):
        """
        Do the query as a single thread for a client.
        :return: None
        """
        logs = []  # the result of query

        # do the query for each host
        with socket(AF_INET, SOCK_STREAM) as s:
            try:
                t_start = time.time()
                s.connect((self.host, self.port))

                # send query pattern as json format
                data = pickle.dumps(self.func_args)
                print('[INFO] sending {} bytes to {}'.format(len(data), self.host))
                s.sendall(data)

                # receive return results
                while True:
                    data = s.recv(4096)
                    if data:
                        msg = pickle.loads(data)
                        print('[INFO] query returns, result: ', msg.content)
                    else:
                        break

                t_end = time.time()
                self.time_cost = t_end - t_start

            # handle the client exception
            except (OSError, error) as e:
                print('[ERROR]: ', self.host, e.__class__().__str__(), e.__str__())


class ClientProtocol(threading.Thread):
    def __init__(self, host,port):
        super(ClientProtocol, self).__init__()
        self.socket = None
        self.host = host
        self.port = port
        self.func_args = []
        self.time_cost = -1

    def connect(self):
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.connect((self.host, self.port))

    def close(self):
        self.socket.shutdown(socket.SHUT_WR)
        self.socket.close()
        self.socket = None

    def send_data(self):
        data = pickle.dumps(self.func_args)

        with open(self.host+"_send_test.txt", "wb") as fp:  # Pickling
            pickle.dump(data, fp)

        # use struct to make sure we have a consistent endianness on the length
        length = pack('>Q', len(data))

        # sendall to make sure it blocks if there's back-pressure on the socket
        self.socket.sendall(length)
        self.socket.sendall(data)

    def run(self):
        try:
            t_start = time.time()
            self.connect()
            self.send_data()
            t_end = time.time()
            self.time_cost = t_end - t_start
        except (OSError, error) as e:
            print('[ERROR]: ', self.host, e.__class__().__str__(), e.__str__())

class Client:
    def __init__(self, hosts=HOSTS, port=PORT):
        self.hosts = hosts
        self.port = port

    def query(self, total_args):
        """
        Do the query as a client. Kill the client after finishing the query.

        :param pattern: query string, e.g. 'a'(raw string), 'a[a-z]b'(regex)
        :return: None
        """

        time_start = time.time()  # record total parallel time
        d_time = {}  # record time cost for each thread

        # assert worker for each query
        workers = [ClientProtocol(host, PORT) for host in HOSTS]
        num_hosts = len(HOSTS)

        print('[INFO] Assign search_space to {} machines'.format(num_hosts))

        for idx, total_arg in enumerate(total_args):
            dst_worker_idx = idx % num_hosts
            workers[dst_worker_idx].func_args.append(total_arg)

        print('[INFO] Start sending parameters ...')

        for worker in workers:
            print('check HOST', worker.host)
            worker.start()

        # end each worker, record time cost
        for worker in workers:
            worker.join()
            d_time[worker.host] = worker.time_cost

        time_end = time.time()  # # record total parallel time
        print('Used %.4f hs.' % ((time_end - time_start) / 3600))


if __name__ == '__main__':
    c = Client()

    usr_configs_template = config.parse_config('./yaml/template.yaml')
    search_space = config.parse_search_space('./yaml/search_space.yaml')
    total_args = [[usr_configs_template, s] for s in search_space]

    c.query(total_args)
