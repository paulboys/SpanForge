# License

## Project License

SpanForge is released under the **MIT License**.

---

## MIT License

```
MIT License

Copyright (c) 2024-2025 SpanForge Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Third-Party Licenses

SpanForge depends on several third-party libraries, each with their own licenses:

### Core Dependencies

#### PyTorch
- **License**: BSD-3-Clause
- **Copyright**: Facebook, Inc. and its affiliates
- **Link**: https://github.com/pytorch/pytorch/blob/master/LICENSE

#### Transformers (HuggingFace)
- **License**: Apache License 2.0
- **Copyright**: The HuggingFace Inc. team
- **Link**: https://github.com/huggingface/transformers/blob/main/LICENSE

#### RapidFuzz
- **License**: MIT License
- **Copyright**: Max Bachmann
- **Link**: https://github.com/maxbachmann/RapidFuzz/blob/main/LICENSE

#### Pydantic
- **License**: MIT License
- **Copyright**: Samuel Colvin
- **Link**: https://github.com/pydantic/pydantic/blob/main/LICENSE

#### Pandas
- **License**: BSD 3-Clause License
- **Copyright**: AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team
- **Link**: https://github.com/pandas-dev/pandas/blob/main/LICENSE

### Development Dependencies

#### pytest
- **License**: MIT License
- **Link**: https://github.com/pytest-dev/pytest/blob/main/LICENSE

#### Black
- **License**: MIT License
- **Link**: https://github.com/psf/black/blob/main/LICENSE

#### isort
- **License**: MIT License
- **Link**: https://github.com/PyCQA/isort/blob/main/LICENSE

### Documentation Dependencies

#### MkDocs
- **License**: BSD 2-Clause License
- **Link**: https://github.com/mkdocs/mkdocs/blob/master/LICENSE

#### MkDocs Material
- **License**: MIT License
- **Copyright**: Martin Donath
- **Link**: https://github.com/squidfunk/mkdocs-material/blob/master/LICENSE

#### mkdocstrings
- **License**: ISC License
- **Link**: https://github.com/mkdocstrings/mkdocstrings/blob/master/LICENSE

---

## Model Licenses

### BioBERT

SpanForge uses **BioBERT** (`dmis-lab/biobert-base-cased-v1.1`) as the default model.

- **License**: Apache License 2.0
- **Authors**: DMIS Lab, Korea University
- **Citation**:
  ```bibtex
  @article{lee2020biobert,
    title={BioBERT: a pre-trained biomedical language representation model for biomedical text mining},
    author={Lee, Jinhyuk and Yoon, Wonjin and Kim, Sungdong and Kim, Donghyeon and Kim, Sunkyu and So, Chan Ho and Kang, Jaewoo},
    journal={Bioinformatics},
    volume={36},
    number={4},
    pages={1234--1240},
    year={2020},
    publisher={Oxford University Press}
  }
  ```
- **Link**: https://github.com/dmis-lab/biobert
- **HuggingFace**: https://huggingface.co/dmis-lab/biobert-base-cased-v1.1

**Note:** When using BioBERT in research, please cite the original paper.

---

## Lexicon Licenses

### MedDRA

The symptom lexicon (`data/lexicon/symptoms.csv`) is derived from **MedDRA** (Medical Dictionary for Regulatory Activities).

- **License**: MedDRA® trademark is registered by the International Federation of Pharmaceutical Manufacturers and Associations (IFPMA) on behalf of the International Council for Harmonisation of Technical Requirements for Pharmaceuticals for Human Use (ICH).
- **Usage**: MedDRA content is subject to licensing agreements. The derived lexicon in SpanForge uses publicly available information and does not redistribute proprietary MedDRA data.
- **Link**: https://www.meddra.org/

**Important:** If you use MedDRA-derived data in production or research, ensure compliance with MedDRA licensing terms.

### Custom Lexicons

Product lexicon (`data/lexicon/products.csv`) is compiled from public sources and does not contain proprietary information.

---

## Data Privacy

### User Data

- **SpanForge does NOT collect or transmit user data**
- **All processing is local** (no external API calls except HuggingFace model downloads)
- **Label Studio telemetry is disabled** (via `LABEL_STUDIO_DISABLE_TELEMETRY=1`)

### Complaints Data

- **Raw complaint texts are NOT included** in the repository
- **Example data is synthetic** or de-identified
- **Users must ensure compliance** with privacy regulations (HIPAA, GDPR) when processing real data

---

## Contributing

By contributing to SpanForge, you agree that your contributions will be licensed under the **MIT License**.

See [Contributing Guide](../development/contributing.md) for details.

---

## Disclaimer

**NO WARRANTY**

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED. THE AUTHORS DISCLAIM ALL WARRANTIES, INCLUDING BUT NOT LIMITED TO MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.

**NO LIABILITY**

IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY ARISING FROM THE USE OF THIS SOFTWARE.

**MEDICAL USE**

THIS SOFTWARE IS FOR RESEARCH PURPOSES ONLY. IT IS NOT INTENDED FOR CLINICAL DIAGNOSIS, TREATMENT, OR REGULATORY DECISION-MAKING. USERS MUST VALIDATE RESULTS INDEPENDENTLY.

---

## Contact

For licensing questions or commercial use inquiries, please contact:

- **GitHub Issues**: [SpanForge Issues](#)
- **Email**: (Add contact email if applicable)

---

*Last updated: January 2025*
