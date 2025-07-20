import React, { useState } from 'react'
import JingleForm from './components/JingleForm'
import JingleResult from './components/ResultScreen'
import Loader from './components/LoadingScreen'

function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-white text-gray-800 p-6">
      <h1 className="text-3xl font-bold mb-4 text-center text-indigo-700">Jingly 🎵</h1>
      <JingleForm setResult={setResult} setLoading={setLoading} />
      {loading && <Loader />}
      {result && !loading && <JingleResult result={result} />}
    </div>
  )
}

export default App
