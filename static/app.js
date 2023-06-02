class Chatbox {
   
    constructor() {
        this.args = {
            // openButton: document.querySelector(".chatbox__button"),
            chatBox: document.querySelector(".chatbox__support"),
            sendButton: document.querySelector(".send__button"),
            micbtn: document.querySelector(".mic"),
        };

        this.state = false;
        this.messages = [];
    }

    display() {
        // const { openButton, chatBox, sendButton, micbtn } = this.args;
        const{chatBox, sendButton,micbtn}=this.args;

        // openButton.addEventListener("click", () => this.toggleState(chatBox));

        sendButton.addEventListener("click", () => this.onSendButton(chatBox));
        micbtn.addEventListener("click", () => this.speak(chatBox));
        // micbtn.addEventListener('result',()=>this.speak.)

        const node = chatBox.querySelector("input");
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox);
            }
        });
    }

    toggleState(chatbox) {
        this.state = !this.state;

        // show or hides the box
        if (this.state) {
            chatbox.classList.add("chatbox--active");
        } else {
            chatbox.classList.remove("chatbox--active");
        }
    }

  
    speak(chatbox) {
        console.log("clickz");
        var m = [];
        m = this.messages;
        console.log(m.length);

        var SpeechRecognition = SpeechRecognition || webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        recognition.lang = "en";
        recognition.onstart = function () {
            console.log("listening");
        };

        recognition.onspeechend = function () {
            console.log("end");
            recognition.stop();

        };

        recognition.onresult = function (event) {
            var transcript = event.results[0][0].transcript.toLowerCase();
            const confidence = event.results[0][0].confidence;
            console.log(transcript);
            console.log(typeof transcript);
            /*if (transcript == "stop") {
                      return;
                  }*/

            let msg1 = { name: "User", message: transcript };
            m.push(msg1);

            var html = "";
            m.slice().reverse().forEach(function (item, index) {
                
                    html +=
                        '<div class="messages__item messages__item--operator">' +
                        item.message +
                        "</div>";
                
            });

            const chatmessage = chatbox.querySelector(".chatbox__messages");
            chatmessage.innerHTML = html;

            let text1 = transcript;
            if (text1 === "") {
                      return;
                  }
            //return transcript;
            console.log("this is text1\n" + text1);
            console.log(typeof text1);
            // this.speak2(transcript);
            //return transcript;

            fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                body: JSON.stringify({ message: text1 }),
                mode: "cors",
                headers: {
                    "Content-Type": "application/json",
                },
            })
                .then((r) => r.json())
                .then((r) => {
                    let msg2 = { name: "Aafhana", message: r.answer };
                    let speech = new SpeechSynthesisUtterance();
                    speech.lang = "en";
                    speech.text = msg2.message;
                    m.push(msg2);

                    window.speechSynthesis.speak(speech);
                    var html = "";
                    m.slice().reverse().forEach(function (item, index) {
                            if (item.name === "Aafhana") {
                                html +=
                                    '<div class="messages__item messages__item--visitor">' +
                                    item.message +
                                    "</div>";
                            }
                            else {
                                html +=
                                    '<div class="messages__item messages__item--operator">' +
                                    item.message +
                                    "</div>";
                            }
                             
                        });

                    //writing to chatbox

                    const chatmessage = chatbox.querySelector(".chatbox__messages");
                    chatmessage.innerHTML = html;

                    
                })//.then closes
                .catch((error) => {
                    console.error("Error:", error);
                    
                });
          
            // read out loud the answer
            let speech = new SpeechSynthesisUtterance();
            let voices = speechSynthesis.getVoices();
            speech.voice = voices[1];

            speech.lang = "en-US";
            speech.text = textToSpeak;

            window.speechSynthesis.speak(speech);
        };
       

        recognition.start();
    }

    onSendButton(chatbox) {
       // console.log("clicked");
        var textField = chatbox.querySelector("input");
        let text1 = textField.value;
        console.log(textField.value);
        if (text1 === "") {
            return;
        }
        let y=text1.length
        //if(y>100)
          
        console.log(y);
        console.log(typeof text1);

        let msg1 = { name: "User", message: text1 };
        this.messages.push(msg1);

        fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: JSON.stringify({ message: text1 }),
            mode: "cors",
            headers: {
                "Content-Type": "application/json",
            },
        })
            .then((r) => r.json())
            .then((r) => {
                let msg2 = { name: "Aafhana", message: r.answer };
                console.log(msg2);
                this.messages.push(msg2);
                console.log(msg2.message);
                this.updateChatText(chatbox);
                textField.value = "";
            })
            .catch((error) => {
                console.error("Error:", error);
                this.updateChatText(chatbox);
                textField.value = "";
            });
    }
    updateChatText(chatbox) {
        var html = "";
        this.messages.slice().reverse().forEach(function (item, index) {
                if (item.name === "Aafhana") {
                   
                    html +=
                        '<div class="messages__item messages__item--visitor">' +
                        item.message +
                        "</div>";
                } else {
                   
                    html +=
                        '<div class="messages__item messages__item--operator">' +
                        item.message +
                        "</div>";
                }
            });

        const chatmessage = chatbox.querySelector(".chatbox__messages");
        chatmessage.innerHTML = html;
    }
}

const chatbox = new Chatbox();
chatbox.display();
