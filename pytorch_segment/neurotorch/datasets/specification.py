from neurotorch.datasets.dataset import (AlignedVolume, Array, PooledVolume)
from neurotorch.datasets.filetypes import (TiffVolume, Hdf5Volume)
from neurotorch.datasets.datatypes import (BoundingBox, Vector)
from abc import (ABC, abstractmethod)
import json
import os


class Spec(ABC):
    """
    An abstract class for specifying a volume dataset structure
    """
    def openVolume(self, volume_spec):
        """
        Opens a volume from a volume specification

        :param volume_spec: A dictionary specifying the volume's parameters

        :return: The volume corresponding to the volume dataset
        """
        try:
            filename = os.path.abspath(volume_spec["filename"])

            if filename.endswith(".tif"):
                edges = volume_spec["bounding_box"]
                bounding_box = BoundingBox(Vector(*edges[0]),
                                           Vector(*edges[1]))
                volume = TiffVolume(filename, bounding_box)

                return volume

            elif filename.endswith(".hdf5"):
                pooled_volume = PooledVolume()
                for dataset in volume_spec["datasets"]:
                    edges = dataset["bounding_box"]
                    bounding_box = BoundingBox(Vector(*edges[0]),
                                               Vector(*edges[1]))
                    volume = Hdf5Volume(filename, dataset, bounding_box)
                    pooled_volume.add(volume)

                return pooled_volume

            else:
                error_string = "{} is an unsupported filetype".format(volume_type)
                raise ValueError(error_string)

        except KeyError:
            error_string = "given volume_spec is corrupt"
            raise ValueError(error_string)

    def create(self, spec, stack_size=33):
        """
        Creates a pooled volume from a volume dataset specification

        :param spec: An array of dictionaries specifying the volume's parameters
        :param stack_size: The maximum number of open volumes

        :return: The pooled volume of the volume dataset
        """
        pooled_volume = PooledVolume(stack_size=stack_size)

        for item in spec:
            volume = self.openVolume(item)
            pooled_volume.add(volume)

        return pooled_volume

    def open(self, spec_filename):
        """
        Opens a pooled volume from a volume dataset specification file

        :param spec_filename: The filename of the volume dataset specification
        :return: The pooled volume of the volume dataset
        """
        spec = self.parse(spec_filename)

        cwd = os.getcwd()
        os.chdir(os.path.dirname(spec_filename))
        pooled_volume = self.create(spec)
        os.chdir(cwd)

        return pooled_volume

    @abstractmethod
    def parse(self, spec_filename):
        """
        Parses a specification file

        :param spec_filename: The filename of the volume dataset specification

        :return: An array of dictionaries specifying the volume's parameters
        """
        pass


class JsonSpec(Spec):
    def parse(self, spec_filename):
        """
        Parses a JSON specification file

        :param spec_filename: The filename of the JSON specification

        :return: An array of dictionaries specifying the volume's parameters
        """
        if not spec_filename.endswith(".json"):
            error_string = "{} is not a JSON file".format(spec_filename)
            raise ValueError(error_string)

        with open(spec_filename, 'r') as f:
            spec = json.load(f)

            return spec
