# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class TerminationCondition:
    def __init__(self, max_depth, end_strings):
        self.end_strings = end_strings
        self.max_depth = max_depth
    
    def ends_by_end_strings(self, text):
        for end_string in self.end_strings:
            if text.endswith(end_string):
                return True
        return False
    
    def ends_by_depth(self, depth):
        return depth >= self.max_depth

    def __call__(self, text, depth):
        if self.ends_by_depth(depth):
            return True
        return self.ends_by_end_strings(text)
