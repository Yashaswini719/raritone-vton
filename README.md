\# Raritone 2D Virtual Try-On



Raritone 2D Virtual Try-On is an AI/ML pipeline designed to generate a virtual try-on result by combining a person image with a garment image.



The project is being developed as Raritone's own modular Virtual Try-On pipeline rather than depending entirely on a ready-made VTON application.



The current implementation focuses on building and integrating the complete processing pipeline:



Person Image + Garment Image

&#x20;       ↓

Image Validation

&#x20;       ↓

Person Processing

&#x20;       ↓

Pose Estimation

&#x20;       ↓

Person Segmentation

&#x20;       ↓

Garment Segmentation

&#x20;       ↓

Garment Alignment

&#x20;       ↓

Try-On Generation

&#x20;       ↓

Output Validation

&#x20;       ↓

Try-On Result





\## Project Status



The project is currently under active development.



\### Completed



\- Person image loading and validation

\- Garment image loading and validation

\- Person segmentation

\- Person mask normalization

\- Garment segmentation

\- Garment mask generation

\- Garment preprocessing

\- Garment alignment pipeline

\- Garment control-point generation

\- Try-on mask generation

\- Alpha-based try-on compositing

\- Try-on output generation

\- Output/debug image generation

\- Automated tests for the try-on pipeline

\- Kaggle GPU environment setup

\- Kaggle-based VTON experimentation

\- FastAPI service

\- Swagger/OpenAPI documentation

\- Health endpoint

\- Try-on API endpoint structure

\- Try-on result endpoint structure



\### Current Work



The remaining integration work is:



\- Connect the FastAPI upload endpoint to the complete VTON inference pipeline

\- Improve garment-to-body alignment

\- Improve garment deformation

\- Improve sleeve and shoulder alignment

\- Improve occlusion handling

\- Improve preservation of the original person's body regions

\- Validate generated results with multiple test cases

\- Complete final evaluation and failure analysis





\## Architecture



```text

&#x20;                   RARITONE VTON

&#x20;                         |

&#x20;            +------------+------------+

&#x20;            |                         |

&#x20;      Person Image              Garment Image

&#x20;            |                         |

&#x20;            v                         v

&#x20;   Image Validation          Image Validation

&#x20;            |                         |

&#x20;            v                         v

&#x20;    Person Processing       Garment Processing

&#x20;            |                         |

&#x20;      +-----+-----+           +-------+-------+

&#x20;      |           |           |               |

&#x20;     Pose      Segmentation  Mask         Alignment

&#x20;      |           |           |               |

&#x20;      +-----------+-----------+---------------+

&#x20;                          |

&#x20;                          v

&#x20;                   VTON Inference

&#x20;                          |

&#x20;                          v

&#x20;                   Try-On Generation

&#x20;                          |

&#x20;                          v

&#x20;                  Output Validation

&#x20;                          |

&#x20;                          v

&#x20;                    Final Result

