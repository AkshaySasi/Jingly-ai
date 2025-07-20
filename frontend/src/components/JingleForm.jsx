import React, { useState } from 'react'
import axios from 'axios'

const JingleForm = ({ setResult, setLoading }) => {
  const [form, setForm] = useState({
    brand_name: '',
    motto: '',
    product_brief: '',
    genre: '',
    reference_audio_path: null,
  })

  const handleChange = (e) => {
    const { name, value, files } = e.target
    setForm(prev => ({
      ...prev,
      [name]: files ? files[0] : value
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setResult(null)

    try {
      const formData = new FormData()
      Object.entries(form).forEach(([key, val]) => {
        if (val) formData.append(key, val)
      })

      const response = await axios.post('http://localhost:8000/generate-jingle', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setResult(response.data)
    } catch (err) {
      console.error("Error:", err)
      alert("Something went wrong. Try again.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="max-w-xl mx-auto bg-white shadow-lg p-6 rounded-lg space-y-4">
      {["brand_name", "motto", "product_brief", "genre"].map(field => (
        <div key={field}>
          <label className="block font-semibold capitalize">{field.replace("_", " ")}</label>
          <input
            type="text"
            name={field}
            required
            value={form[field]}
            onChange={handleChange}
            className="w-full px-4 py-2 border rounded-md"
          />
        </div>
      ))}
      <div>
        <label className="block font-semibold">Reference Audio (optional)</label>
        <input type="file" name="reference_audio_path" accept="audio/*" onChange={handleChange} />
      </div>
      <button type="submit" className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700">
        Generate Jingle
      </button>
    </form>
  )
}

export default JingleForm