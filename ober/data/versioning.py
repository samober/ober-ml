import os

class VersionedFile:
	
	"""
	
	Creates a file system for storing different versions of the same type of data.
	
	This is used to ensure that data resulting from intermediate processing and other important information can be recovered in the case of corruption.
	
	It also allows multiple stages of processing/learning to be easily see when new data is available and train on it.
	
	The directory has a structure as follows:
		
	.. code-block:: python
	
		<BASE_PATH>
			<VERSION_ID>
				<DATA_FILE_1>
				<DATA_FILE_2>
				[...]
			<VERSION_ID>
				<DATA_FILE_1>
				<DATA_FILE_2>
				[...]
			[...]
	
	**Arguments:**
	
		* path *(str)* - The path to the main directory for this versioned file system.
		* version_num_length *(int)* - The length of the version numbers for the directory files. Defaults to 5.
	
	"""
	
	def __init__(self, path, version_num_length=5):
		# path to base directory
		self.path = path
		# length of version number string
		self.version_num_length = version_num_length
		
		# ensure directory path exists
		self._ensure_directory_path()
		
		# current versions in directory (set)
		self.versions = self._get_versions_list()
		
	def _ensure_directory_path(self):
		if not os.path.exists(self.path):
			os.makedirs(self.path)
			
	def _ensure_version_path(self, version):
		path = self.get_version_path(version)
		if not os.path.exists(path):
			os.makedirs(path)
			return True
		return False
		
	def _get_versions_list(self):
		"""
		Returns a list of all the current versions in the directory (as integers)
		"""
		versions = set()
		# list all paths in directory
		for f in os.listdir(self.path):
			try:
				# append if an integer
				versions.add(int(f))
			except:
				continue
		return versions
			
	def get_versions(self):
		"""
		
		Returns a list of all the versions currently available for this file system.
		
		:return: A list of version numbers.
		:rtype: ``List[int]``
		
		"""
		return list(self.versions)
			
	def get_version_path(self, version):
		"""
		
		Gets the path to the directory for the specific version.
		
		:param version: The version number for the directory.
		:type version: int
		:return: The path to the version's directory.
		:rtype: str
		
		"""
		return os.path.join(self.path, "{:0{length}}".format(version, length=self.version_num_length))
		
	def check_version_path(self, version):
		"""
		
		Checks if the current version's directory currently exists.
		
		:param version: The version number to check for.
		:type version: int
		:return: True if the path does exists, False if it does not
		:rtype: bool
		
		"""
		return os.path.exists(self.get_version_path(version))
		
	def get_latest_version(self):
		"""
		
		Returns the latest available version number for this file system.
		
		:return: The latest version number. (or 0 if none were found)
		:rtype: int
		
		"""
		if len(self.versions) > 0:
			return max(self.versions)
		return 0
		
	def create_version(self, version):
		"""
		
		Creates a new version for the data if one does not already exist.
		
		:param version: The version number for the data. (must be >= 1)
		:type version: int
		
		"""
		if version >= 1 and self._ensure_version_path(version):
			self.versions.add(version)
			
	def get_file_path(self, version, filename, ignore_exists=False):
		"""
		
		Return the direct path to a file in a specific version directory.
		
		:param version: The version of the file system to look in.
		:type version: int
		:param filename: The name of the data file in the specific version's directory.
		:type filename: str
		:param ignore_exists: If True, return the path whether or not it exists.
		:type ignore_exists: bool
		:return: The path to the versioned file. (or None if the version does not exist)
		:rtype: str
		
		"""
		if ignore_exists or self.check_version_path(version):
			return os.path.join(self.get_version_path(version), filename)
		return None
		
	def get_file_paths(self, filename):
		"""
		
		Return a list of the direct paths to a data file for all versions in the file system.
		
		:param filename: The name of the data file to retrieve.
		:type filename: str
		:return: The paths to the data file in the form of tuples: (version_num, file_path)
		:rtype: ``List[Tuple[int, str]]``
		
		"""
		file_paths = []
		for version in self.versions:
			file_paths.append((version, self.get_file_path(version, filename)))
		return file_paths
		
	def create_latest_version(self):
		"""
		
		Create a new version and make it the latest.
		
		The new version number will be the current latest version incremented by one. If there are currently no versions in the file system it will start at version 1.
		
		:return: The new version number for the version just created.
		:rtype: int
		
		"""
		new_version = self.get_latest_version() + 1
		self.create_version(new_version)
		return new_version
		
	def get_latest_file_path(self, filename, ignore_exists=False):
		"""
		
		Return the direct path to a file for the latest version in the directory.
		
		:param filename: The name of the data file.
		:type filename: str
		:param ignore_exists: If True, return the path whether or not it exists.
		:type ignore_exists: bool
		:return: The path to the versioned file. (or None if there are no versions available)
		:rtype: str
		
		"""
		return self.get_file_path(self.get_latest_version(), filename, ignore_exists=ignore_exists)