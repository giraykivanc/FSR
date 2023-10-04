
<a name="readme-top"></a>





## Overview
This program is designed for training and working with three different machine learning network models in PyTorch: "Base," "Block1," and "Block2." This README provides an overview of the program's functionality and usage.



<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* Pytorch

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Prerequisites

Before using this program, make sure you have the following prerequisites installed:
* Python (3.x recommended)
* PyTorch

You can install the required libraries using pip:

* pip
  ```sh
  pip3 install torch torchvision torchaudio
  ```


<!-- USAGE EXAMPLES -->
## Usage

To run the program, use the following command:

* bash
  ```sh
  python test.py --model <model_name> --freeze <0 or 1>
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Command-line Arguments

* --model: Specifies the machine learning model to use. You can choose from "base," "block1," or "block2." The default value is "base."

* --freeze: Specifies whether to freeze the weights of Block1 during training of Block2. Use "0" to disable freezing (default) or "1" to enable freezing.


<!-- ROADMAP -->
## Examples

1- To train the "Base" model without freezing Block1:
* bash
  ```sh
  python test.py --model base
  ```

2- To train the "Block1" model without freezing Block1:
* bash
  ```sh
  python test.py --model block1
  ```

3- To train the "Block2" model with Block1's weights frozen:
* bash
  ```sh
  python test.py --model block2 --freeze 1
  ```

## Customization

You can customize the behavior of the program by modifying the source code in the following files:

* test.py: The main script that accepts user inputs and manages the training process.
* BaseNetwork.py: Contains the definition and training logic for the "Base" model.
* Block1.py: Contains the definition and training logic for the "Block1" model.
* Block2.py: Contains the definition and training logic for the "Block2" model.

Feel free to explore and modify these files to suit your specific needs.


<!-- LICENSE -->
## License

Distributed under the MIT License.

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
[Pytorch-url]: https://pytorch.org/ 
