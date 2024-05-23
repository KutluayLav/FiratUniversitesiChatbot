import React, { useState, useEffect, useRef } from 'react';
import { MdOutlineSend } from 'react-icons/md';
import logo from './assests/16329166371.png';

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    if(messagesEndRef.current)
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = () => {
    if (input.trim()) {
      setMessages([...messages, { text: input, sender: 'user' }]);
      setInput('');
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-400">
      <div className="flex flex-col w-full max-w-4xl h-[70vh] bg-white shadow-lg rounded-lg overflow-hidden">
        <div className="flex items-center p-4 bg-red-500">
          <img src={logo} alt="Logo" className="w-10 h-10 rounded-full mr-3" />
          <h1 className="text-white text-lg">Fırat Üniversitesi Yardımcı Botu</h1>
        </div>
        <div className="flex-grow p-8 overflow-y-auto">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`mb-4 p-2 max-w-xs rounded-lg ${
                message.sender === 'user'
                  ? 'bg-gray-300 text-black self-end ml-auto'
                  : 'bg-red-500 text-white mr-auto'
              }`}
            >
              {message.text}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        <div className="p-4 bg-gray-100 flex">
          <input
            className="flex-grow p-2 rounded-lg border border-slate-300 focus:outline-none focus:border-red-500 focus:ring-red-500 focus:ring-1 sm:text-sm"
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Mesajınızı yazın..."
          />
          <button
            className="ml-2 p-2 bg-red-500 text-white rounded-lg flex items-center justify-center hover:bg-red-600 hover:text-gray-100 transition-colors duration-300 ease-in-out"
            onClick={sendMessage}
            >
            <MdOutlineSend size={24} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
