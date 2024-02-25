# ------------------------------------------------------------------------
# Copyright (c) 2023-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Tokenize Anything via Prompting."""

from tokenize_anything.models import model_registry
from tokenize_anything.predictor import TapPredictor
from tokenize_anything.automatic_mask_generator import TapAutomaticMaskGenerator
# from tokenize_anything.version import __version__
