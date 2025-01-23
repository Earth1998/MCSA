import sys
import datetime
 
class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='utf-8')
        self.previousMsg = None
        sys.stdout = self
 
    def write(self, message):
        if self.previousMsg == None or "\n" in self.previousMsg:
            topMsg = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " : "
            self.terminal.write(topMsg) # terminal
            self.log.write(topMsg) # output log
 
        if isinstance(message, str):
            self.previousMsg = message
        if self.previousMsg == None:
            self.previousMsg = ""
 
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # flush
 
    def flush(self):
        pass
 
