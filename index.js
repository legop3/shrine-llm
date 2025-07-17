const { spawn } = require('node:child_process')
const { Client, Events, GatewayIntentBits, ChatInputCommandInteraction } = require('discord.js')
const EventEmitter = require('events')
require('dotenv').config()
// console.log()
// const input_text = 'I am noah, '
// const max_length = 50
// const temp = 0.1

// const bridge = spawn('python', ['generate.py', input_text || 'you are a tree', max_length, temp])

// bridge.stdout.on('data', (data) => {
//     console.log(data.toString())
// })

// bridge.stderr.on('data', (data) => {
//     console.error('stderr:', data.toString())
// })

// bridge.on('close', (code) => {
//     console.log(`Process exited with code ${code}`)
// })

// bridge.on('error', (err) => {
//     console.error('Failed to start process:', err)
// })

client = new Client({ intents: [GatewayIntentBits.MessageContent, GatewayIntentBits.GuildMessages, GatewayIntentBits.Guilds]})



class Generator extends EventEmitter {
    constructor(input, length, temperature, path) {
        super()
        this.input = input,
        this.length = length,
        this.temperature = temperature
        this.path = path

        this.bridge = spawn('python', ['generate.py', input || 'you are a tree', length, temperature, path])

        this.bridge.stdout.on('data', (data) => {
            // console.log(data)
            this.emit('token', data.toString())
        })

        this.bridge.stderr.on('data', (err) => {
            console.error(err.toString())
        })

        this.bridge.on('close', () => {
            console.log('bridge closed')
            this.emit('close')
        })

    }



}


client.on('ready', readyClient => {
    console.log(readyClient.user.tag)
})

client.on('messageCreate', async message => {
    if(message.author.bot) {return}
    // console.log(message.content)
    const prefix = message.content.toLowerCase().split(' ')[0]
    const content = message.content.split(' ').slice(1).join(' ')
    // console.log(content)
    switch (prefix) {
        case 'shrine':
            // console.log(message.author)
            // message.reply(`shrine speak ${content}`)
            streaming = ''
            input = `${message.author.globalName}: ${content}\n`
            let shrinegen = new Generator(input, 1000, 0.7, './discord_finetuned_model')
            toEdit = await message.reply('Processing Information')
            // console.log(toEdit)
            
            lastEdit = 0
            editCooldown = 2500
            pendingEdit = false

            flushEdit = () => {
            if (streaming && !pendingEdit) {
                pendingEdit = true

                if(streaming) {
                    toEdit.edit(`${streaming}`)
                }

                lastEdit = Date.now()
                pendingEdit = false
            }
            }

            shrinegen.on('token', token => {
                console.log(token)
                streaming += token
                console.log(streaming)
                
                const now = Date.now()
                if (now - lastEdit > editCooldown) {
                    flushEdit()
                }
            })

            // Handle stream completion - flush any remaining tokens
            shrinegen.on('close', () => {
            // Force final edit regardless of cooldown
                flushEdit()
            })



            break;
        case 'ross':
            // message.reply('Ross is not yet implemented')

            // message.reply(`shrine speak ${content}`)
            streaming = ''
            input = `${message.author.globalName}: ${content}\n`
            let rossgen = new Generator(input, 1000, 1, './ross_model')
            toEdit = await message.reply('Processing Information')
            // console.log(toEdit)
            
            lastEdit = 0
            editCooldown = 2500
            pendingEdit = false

            flushEdit = () => {
            if (streaming && !pendingEdit) {
                pendingEdit = true

                if(streaming) {
                    toEdit.edit(`${streaming}`)
                }

                lastEdit = Date.now()
                pendingEdit = false
            }
            }

            rossgen.on('token', token => {
                console.log(token)
                streaming += token
                console.log(streaming)
                
                const now = Date.now()
                if (now - lastEdit > editCooldown) {
                    flushEdit()
                }
            })

            // Handle stream completion - flush any remaining tokens
            rossgen.on('close', () => {
            // Force final edit regardless of cooldown
                flushEdit()
            })
            




            break;
    
        default:
            break;
    }
})





client.login(process.env.DISCORD_BOT_TOKEN)