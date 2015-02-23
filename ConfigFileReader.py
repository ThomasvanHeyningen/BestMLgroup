import ConfigParser

class ConfigFileReader:

	def __init__(self):
		self.config_reader = ConfigParser.ConfigParser()
		self.config_reader.read("configFile.ini")

	def ConfigSectionMap(self,section):
	    dict1 = {}
	    options = self.config_reader.options(section)
	    for option in options:
		try:
		    dict1[option] = self.config_reader.get(section, option)
		    if dict1[option] == -1:
		        DebugPrint("skip: %s" % option)
		except:
		    print("exception on %s!" % option)
		    dict1[option] = None
	    return dict1

	def getVariable(self, section, variable):
		return (self.ConfigSectionMap(section)[variable])

