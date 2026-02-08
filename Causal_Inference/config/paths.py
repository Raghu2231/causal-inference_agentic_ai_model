from pathlib import PurePath, PurePosixPath
from typing import Optional
import datetime
import subprocess
from .archivist import ARCHIVIST_HDFS_OUTPUT_DIR

from .. import logger


class PathConfig:
    '''
    Contains logic for local file structure.
    '''

    def __init__(
        self,
        base_path: str = ARCHIVIST_HDFS_OUTPUT_DIR,
        run_id: Optional[str] = None
    ) -> None:
        '''
        :param base_path: A path to use for the file structure. Defaults to
        OUTPUT_DIR
        :param run_id: An optional, string run ID, for loading and using a sub-
        folder.

        '''
        sanitized_base_path: PurePath = HDFSPath(base_path)
        self.run_id: str = (
            run_id
            if run_id is not None else
            f'{datetime.utcnow():%Y%m%d-%H%M%S}-' + str(uuid4())[:8]
        )

        self.run_folder = sanitized_base_path / self.run_id
        self.run_folder.mkdir(exist_ok=True, parents=True)
        self.log: PurePath = self.run_folder / 'suggestions_measurement.log'

        # /path/to/{{base_path}}/{{run_id}}
        # TODO: Add path for ModelConfig JSON object
    
        # /path/to/{{base_path}}/{{run_id}}/model_config/
        self.model_config: PurePath = self.run_folder / 'model_config'
        self.model_config.mkdir(exist_ok=True, parents=True)

        # /path/to/{{base_path}}/{{run_id}}/models/
        self.models: PurePath = self.run_folder / 'models'
        self.models.mkdir(exist_ok=True, parents=True)

        self.data: PurePath = self.run_folder / 'data'
        self.data.mkdir(exist_ok=True, parents=True)

        self.refinery: PurePath = self.run_folder / 'refinery'
        self.refinery.mkdir(exist_ok=True, parents=True)

        self.viz: PurePath = self.run_folder / 'viz'
        self.viz.mkdir(exist_ok=True, parents=True)
        
        logger.debug(
            'Suggestions Measurement file structure initialized in run folder: %s', 
            self.run_folder
        )

    def model_write_path(self, *, model_type: str, model_id: str, create: bool = True) -> PurePath:
        '''
        Returns the path to write a model to.

        :param model_type: The type of model to write.
        :param model_id: The unique identifier for the model.
        '''
        directory = self.models / f'{model_type}-{model_id}'
        if create:
            directory.mkdir(exist_ok=True, parents=True) 
        return directory

    def data_write_path(self, *, model_type: str, model_id: str, create: bool = True) -> PurePath:
        '''
        Returns the path to write data to.

        :param model_type: The type of model to write.
        :param model_id: The unique identifier for the model.
        '''
        directory = self.data / f'{model_type}-{model_id}'
        if create:
            directory.mkdir(exist_ok=True, parents=True)
        return directory
    
    def model_config_write_path(self, *, model_type: str, create: bool = True) -> PurePath:
        '''
        Returns the path to write model_config to.

        :param model_type: The type of model to write.
        '''
        directory = self.model_config / f'{model_type}'
        if create:
            directory.mkdir(exist_ok=True, parents=True)

        return directory

    def refinery_write_path(self, *, create: bool = True) -> PurePath:
        '''
        Returns the path to write refinery to.
        '''
        directory = self.refinery
        if create:
            directory.mkdir(exist_ok=True, parents=True)
        return directory


class HDFSPath(PurePosixPath):
    __ls_args = ['hdfs', 'dfs', '-ls']
    __mkdir_args = ['hdfs', 'dfs', '-mkdir']
    __mkdir_parents_args = ['hdfs', 'dfs', '-mkdir', '-p']
    __test_dir_args = ['hdfs', 'dfs', '-test', '-d']
    __test_file_args = ['hdfs', 'dfs', '-test', '-e']

    def _run(self, *args, **kwargs):
        ''' This is wrapper around subprocess.run with sane arguments are prespecified. '''
        default_kwargs = {
            "capture_output": True,  # we want to see the stdout and stderr
            "shell": False,  # no fancy shell expansion - it messes with hdfs
            "check": True,  # raise exception if return code is not 0
        }
        return subprocess.run(
            *args,
            **{**default_kwargs, **kwargs}
        )

    def _exists_file(self):
        file_exists_test = self._run(
            self.__test_file_args + [str(self)],
            capture_output=False,
            check=False,
        )
        if file_exists_test.returncode == 0:
            return True
        return False

    def _exists_dir(self):
        file_exists_test = self._run(
            self.__test_dir_args + [str(self)],
            capture_output=False,
            check=False,
        )
        if file_exists_test.returncode == 0:
            return True
        return False

    def exists(self, *, follow_symlinks=True):
        return self._exists_dir() or self._exists_file()

    def mkdir(self, *, follow_symlinks=True, parents=False, exist_ok=False):
        # Check the exist_ok arg first
        if exist_ok and self._exists_dir():
            return self
        if exist_ok and self._exists_file():
            raise FileExistsError(f'File exists: {self}')
        if not exist_ok and self.exists():
            raise FileExistsError(f'Exists: {self}')

        if parents:
            mkdir_args = self.__mkdir_parents_args
        else:
            mkdir_args = self.__mkdir_args
        
        self._run(
            mkdir_args + [str(self)],
        )
        return self

    def ls(self):
        ls = self._run(
            self.__ls_args + [str(self)]
        )

        print(ls.stdout.decode('utf-8'))