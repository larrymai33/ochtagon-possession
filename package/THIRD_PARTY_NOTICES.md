# Third-party notices

The exact upstream competition LICENSE is retained as LICENSE, including its
scope statement. This package incorporates Metrica Sports source and derived
solution code. The retained notice mentions public GT in the upstream repository;
no GT is present in this package. It grants no rights over restricted match assets.

Training used SkillCorner Open Data at revision
`4340d274572876239c154c90bc507a9b3250a656`. Its notice is reproduced below.
No raw SkillCorner data is bundled. This notice is attribution and the observed
source license, not a claim that all possible model-distribution rights have
received an independent legal review. Public distribution remains pending.

## SkillCorner notice

The MIT License

Copyright (c) 2020 SkillCorner. https://skillcorner.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## Runtime packages

Third-party binary wheels are downloaded from the official origins listed in
`repro/runtime-windows-cp311.wheels.json`; they are not vendored in this Git
package. Their distributions carry their own license files and notices. The
nineteen-package hash lock identifies the exact distribution artifacts, including
the CUDA-enabled Torch build, without changing their licenses.
