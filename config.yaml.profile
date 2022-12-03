baseURL: "https://aadi-blogs.web.app/"
languageCode: "en-us"
title: "Hugo Profile"
theme: hugo-profile

Paginate: 9
disqusShortname: https-aadi-blogs-web-app

Menus:
  main:
    - identifier: "blog"
      name: blog
      title: blog
      url: /blog

    - identifier: code
      title: code
      name: code
      url: /code

params:
  framed: true
  title: "Aadidev Sooknanan"
  favicon: /favicon.png
  animate: true
  centerTheme: true
  themeColor: green
  theme:
    defaultThem: "dark"
  hero:
    enable: true
    intro: "Hi, my name is"
    title: "Aadidev"
    subtitle: "I breathe data"
    content: "I'm a data scientist who's worked on million-dollar projects, put machine-learning models into production and contributed to deep learning research"
    image: https://imgur.com/iHR9Ubq.png
    button:
      enable: true
      name: "Resumé"
      url: "https://drive.google.com/file/d/1cGihNgZQo_pdVjigf7tJ6LxPu7Y94_ru/view?usp=share_link"
      download: true
    socialLinks:
      fontAwesomeIcons:
        - icon: fab fa-github
          url: https://github.com/aadi350
        - icon: fab fa-twitter
          url: https://twitter.com/__aadiDev__
        - icon: fab fa-linkedin
          url: https://www.linkedin.com/in/aadidev-sooknanan/

  contact:
    enable: true
    title: "contact"
    content: My inbox is always open. Whether you have a question or just want to say hi, I’ll try my best to get back to you!
    email: aadidevsooknanan@protonmail.com
    btnName: Mail me
