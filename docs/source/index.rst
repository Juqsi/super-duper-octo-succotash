.. super-duper-octo-succotash documentation master file, created by
   sphinx-quickstart on Tue Mar 25 12:14:11 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PlantAI
=======

**PlantAI** is an AI-powered system for automatic plant classification and information retrieval,
designed for home gardeners, hobby botanists, and developers in the smart gardening space.

Using machine learning, PlantAI reliably identifies plant species from images, instantly provides
relevant information, and enables integration with modern IoT applications for automated plant care.

Goals & Motivation
------------------
Many plant owners lack the time or expertise to ensure proper care for their plants.
PlantAI addresses this challenge through intuitive image analysis, combined with a centralized
plant database accessible via API – eliminating the need for time-consuming manual research.

Core Features
-------------
- **Image-based plant identification using trained AI models**
- **Access to plant information via API** from Wikipedia and a custom database ("PlantBase")
- **Modern, intuitive web interface**
- **Modular, maintainable backend architecture**
- **Easily fine-tunable machine learning models**

System Architecture
-------------------
The system is composed of the following core components:

- **Backend**: FastAPI-based, modular, and easy to maintain
- **Frontend**: A Vue-based web application with a clean and intuitive UI
- **AI Module**: Pre-trained classification models with support for fine-tuning
- **PlantBase**: A structured plant information database accessible through a RESTful API

Further Information
-------------------
Project Website: https://web.ase.juqsi.de


Project Setup and Introduction
==============================

This project consists of two Docker containers:

- **Frontend:** Responsible for the user interface, accessible on port 443.
- **Backend:** Handles the API and business logic, accessible on port 8000.

Both containers are defined in a single Docker Compose file and will automatically restart (``unless-stopped``) if they exit unexpectedly.

Setup Instructions
------------------

1. **Prerequisites:**

   - Docker and Docker Compose must be installed.
   - A valid JWT for the PlantBase API must be available and will be used in place of ``<Token>``.
   - (Optional) You can adjust the environment variables:

     - **MIN_ACC:** Specifies the minimum probability (in percent) required for a plant to be returned. The default is 40%.
     - **HOST:** Defaults to ``localhost``.

2. **Project Structure:**

   Create the following directories, where the respective Dockerfiles and source code will reside:

   - ``./frontend``
   - ``./backend``

3. **Environment Variables Setup (in the Backend):**

   - Replace ``<Token>`` in the ``PLANT_API_KEY`` environment variable with your valid JWT.
   - Adjust the ``MIN_ACC`` variable if needed. Remember that the default value is 40% unless specified otherwise.
   - The ``HOST`` variable can also be customized; by default, it is set to ``localhost``.

4. **Starting the Project:**

   Navigate to the directory containing the ``docker-compose.yml`` file and run the following command:

   .. code-block:: bash

      docker-compose up --build

   After starting, the frontend service will be accessible over HTTPS at:

   `https://localhost <https://localhost>`_

   Since a self-signed certificate is used, your browser might warn you about an insecure connection. You can proceed by accepting the warning.

5. **Additional Notes:**

   - Check the logs of both containers to ensure that all services have started correctly.
   - If you make changes to the source code of any service, restart the containers to apply the updates.


.. toctree::
   :maxdepth: 2
   :caption: Modules:

   _modules/backend/index
   _modules/ai_training/index

