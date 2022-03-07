from neurotorch.datasets.dataset import Volume, Array, Data
from neurotorch.datasets.datatypes import BoundingBox, Vector
from abc import abstractmethod
import fnmatch
import os.path
import h5py
import numpy as np
import tifffile as tif


class TiffVolume(Volume):
    def __init__(self, tiff_file, bounding_box: BoundingBox,
                 iteration_size: BoundingBox=BoundingBox(Vector(0, 0, 0),
                                                         Vector(128, 128, 32)),
                 stride: Vector=Vector(64, 64, 16)):
        """
        Loads a TIFF stack file or a directory of TIFF files and creates a
corresponding three-dimensional volume dataset
        :param tiff_file: Either a TIFF stack file or a directory
containing TIFF files
        :param chunk_size: Dimensions of the sample subvolume
        """
        # Set TIFF file and bounding box
        self.setFile(tiff_file)
        super().__init__(bounding_box, iteration_size, stride)

    def setFile(self, tiff_file):
        if os.path.isfile(tiff_file) or os.path.isdir(tiff_file):
            self.tiff_file = tiff_file
        else:
            raise IOError("{} was not found".format(tiff_file))

    def getFile(self):
        return self.tiff_file

    def get(self, bounding_box):
        return self.getArray().get(bounding_box)

    def __enter__(self):
        if os.path.isfile(self.getFile()):
            try:
                print("Opening {}".format(self.getFile()))
                array = tif.imread(self.getFile())

            except IOError:
                raise IOError("TIFF file {} could not be " +
                              "opened".format(self.getFile()))

        elif os.path.isdir(self.getFile()):
            tiff_list = os.listdir(self.getFile())
            tiff_list = filter(lambda f: fnmatch.fnmatch(f, '*.tif'),
                               tiff_list)

            if tiff_list:
                array = tif.TiffSequence(tiff_list).asarray()

        else:
            raise IOError("{} was not found".format(self.getFile()))

        array = Array(array, bounding_box=self.getBoundingBox(),
                      iteration_size=self.getIterationSize(),
                      stride=self.getStride())
        self.setArray(array)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.setArray(None)

    def _indexToBoundingBox(self, idx):
        return self.getArray()._indexToBoundingBox(idx)


class Hdf5Volume(Volume):
    def __init__(self, hdf5_file, dataset, bounding_box: BoundingBox,
                 iteration_size: BoundingBox=BoundingBox(Vector(0, 0, 0),
                                                         Vector(128, 128, 20)),
                 stride: Vector=Vector(64, 64, 10)):
        """
        Loads a HDF5 dataset and creates a corresponding three-dimensional
volume dataset

        :param hdf5_file: A HDF5 file path
        :param dataset: A HDF5 dataset name
        :param chunk_size: Dimensions of the sample subvolume
        """
        self.setFile(hdf5_file)
        self.setDataset(dataset)
        super().__init__(bounding_box, iteration_size, stride)

    def setFile(self, hdf5_file):
        self.hdf5_file = hdf5_file

    def getFile(self):
        return self.hdf5_file

    def setDataset(self, hdf5_dataset):
        self.hdf5_dataset = hdf5_dataset

    def getDataset(self):
        return self.hdf5_dataset

    def __enter__(self):
        if os.path.isfile(self.getFile()):
            with h5py.File(self.getFile(), 'r') as f:
                array = f[self.getDataset()].value
                array = Array(array, bounding_box=self.getBoundingBox(),
                              iteration_size=self.getIterationSize(),
                              stride=self.getStride())
                self.setArray(array)

    def __exit__(self, exc_type, exc_value, traceback):
        self.setArray(None)
