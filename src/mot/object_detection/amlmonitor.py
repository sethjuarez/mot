import re
import operator
import tensorflow as tf
from azureml.core.run import Run
from tensorpack.callbacks import MonitorBase

class AMLMonitor(MonitorBase):
    """
    Print scalar data into terminal.
    """

    def __init__(self, enable_step=False, enable_epoch=True):
        """
        Args:
            enable_step, enable_epoch (bool): whether to print the
                monitor data (if any) between steps or between epochs.
            whitelist (list[str] or None): A list of regex. Only names
                matching some regex will be allowed for printing.
                Defaults to match all names.
            blacklist (list[str] or None): A list of regex. Names matching
                any regex will not be printed. Defaults to match no names.
        """

        self.run = Run.get_context()
        self.offline = self.run.id.startswith('OfflineRun')
        if self.offline:
            print('Offline AML Context: {}'.format(self.offline))
        else:
            print('Online AML Context: {}'.format(self.run))

        self._enable_step = enable_step
        self._enable_epoch = enable_epoch
        self._dic = {}

    # in case we have something to log here.
    def _before_train(self):
        self._trigger()

    def _trigger_step(self):
        if self._enable_step:
            if self.local_step != self.trainer.steps_per_epoch - 1:
                # not the last step
                self._trigger()
            else:
                if not self._enable_epoch:
                    self._trigger()
                # otherwise, will print them together

    def _trigger_epoch(self):
        if self._enable_epoch:
            self._trigger()

    def process_scalar(self, name, val):
        self._dic[name] = float(val)

    def _trigger(self):
        if not self.offline:
            for k, v in sorted(self._dic.items(), key=operator.itemgetter(0)):
                self.run.log(k, v)
            self._dic = {}