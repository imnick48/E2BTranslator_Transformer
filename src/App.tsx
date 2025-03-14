import axios from 'axios';
import React, { useEffect, useState } from 'react';

interface Message {
  id: number;
  english: string;
  bengali: string;
}

const App: React.FC = () => {
  const [data, setData] = useState<Message[]>([]);
  
  const handlebuttonclick = () => {
    const inputElement = document.querySelector('input') as HTMLInputElement;
    const input = inputElement.value;
    axios.post("http://127.0.0.1:5000/translate", {
      english: input
    })
      .then((response) => {
        const newMessage: Message = {
          id: Date.now(),
          english: input,
          bengali: response.data.bengali
        };
        setData(prevData => [...prevData, newMessage]);
        inputElement.value = '';
      })
      .catch((error) => {
        console.log(error);
      });
  };

  useEffect(() => {
    axios.get("http://127.0.0.1:5000/translatedata")
      .then((response) => {
        setData(response.data);
      })
      .catch((error) => {
        console.log(error);
      })
  }, [])
  return (
    <div className="font-sans flex flex-col max-w-xl mx-auto my-10 border border-gray-300 rounded-2xl shadow-lg bg-white overflow-hidden">
      {/* Header */}
      <header className="bg-purple-600 text-white text-center p-4 rounded-t-2xl">
        <h1 className="text-xl font-semibold">English to Bengali</h1>
        <p className="text-sm text-purple-200">Powered by a Transformer model</p>
      </header>

      {/* Chat Area */}
      <div className="p-4 flex flex-col gap-3 min-h-[600px] bg-gray-50">
        {data.map(message => (
          <div key={message.id} className="flex flex-col gap-2 pb-3">
            <div className="bg-gray-200 text-black px-4 py-3 rounded-2xl max-w-[75%] text-sm leading-relaxed self-start">
              {message.english}
            </div>
            <div className="bg-purple-600 text-white px-4 py-3 rounded-2xl max-w-[75%] text-sm leading-relaxed self-end">
              {message.bengali}
            </div>
          </div>
        ))}
      </div>

      {/* Input Area */}
      <div className="flex items-center p-3 border-t border-gray-200 bg-gray-100">
        <input
          type="text"
          placeholder="Type in English..."
          className="flex-1 p-2 text-sm border border-gray-300 outline-none rounded-[20px] px-4"
        />
        <button className="ml-2 px-4 py-2 bg-purple-600 text-white text-lg rounded-full" onClick={handlebuttonclick}>➤</button>
      </div>
    </div>
  );
};

export default App;
