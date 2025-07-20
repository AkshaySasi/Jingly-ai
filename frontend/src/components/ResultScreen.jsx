import React from 'react'

const JingleResult = ({ result }) => {
  return (
    <div className="mt-8 text-center">
      <h2 className="text-xl font-semibold text-green-700 mb-2">Jingle Generated ✅</h2>
      <p className="mb-4 text-gray-700 italic">Prompt used: <span className="font-mono">{result.musicgen_prompt}</span></p>
      <audio controls className="mx-auto">
        <source src={`http://localhost:8000/${result.output_path}`} type="audio/wav" />
        Your browser does not support audio playback.
      </audio>
      <br />
      <a href={`http://localhost:8000/${result.output_path}`} download className="inline-block mt-4 text-indigo-600 underline">
        ⬇️ Download Jingle
      </a>
    </div>
  )
}

export default JingleResult
